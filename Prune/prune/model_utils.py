from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import block_influence


def get_nested_attr(root, dotted_path: str):
    obj = root
    for name in dotted_path.split("."):
        obj = getattr(obj, name)
    return obj


def remove_layers(model, layers_to_remove: Iterable[int], layers_path: str = "model.layers") -> List[int]:
    """Delete transformer blocks and refresh attention layer indices when present."""
    layers = get_nested_attr(model, layers_path)
    removed = []
    for layer_idx in sorted(set(int(i) for i in layers_to_remove), reverse=True):
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"Layer {layer_idx} is outside the available range 0..{len(layers) - 1}")
        del layers[layer_idx]
        removed.append(layer_idx)

    for new_idx, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "layer_idx"):
            self_attn.layer_idx = new_idx
    return sorted(removed)


class ShortGPTModel:
    """Small HF wrapper used for ShortGPT BI and TAPPA q-similarity collection."""

    def __init__(
        self,
        model_name_or_path: str,
        layers_path: str = "model.layers",
        device: str = "cuda",
        dtype: str = "float16",
        trust_remote_code: bool = True,
        token: Optional[str] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            token=token,
            trust_remote_code=trust_remote_code,
        ).to(device)
        self.model.eval()
        self.layers = get_nested_attr(self.model, layers_path)
        self.device = device
        self.bi_scores = np.zeros(len(self.layers), dtype=np.float64)
        self.bi_samples = 0
        self.q_scores = np.zeros(len(self.layers), dtype=np.float64)
        self.q_samples = 0

    @torch.inference_mode()
    def eval_bi(
        self,
        prompts: List[str],
        max_seq_len: int,
        stride: int = 256,
        max_gen_len: int = 0,
        angular: bool = False,
    ) -> None:
        prompt_tokens = self.tokenizer(
            prompts,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = prompt_tokens.input_ids
        attn_mask = prompt_tokens.attention_mask
        max_prompt_len = max(len(t) for t in input_ids)

        for start in range(0, max_prompt_len, stride):
            seq_ids = (attn_mask.sum(dim=-1) > start).nonzero().squeeze()
            seq_ids = seq_ids.unsqueeze(0) if seq_ids.dim() == 0 else seq_ids
            inputs = input_ids[seq_ids, start : start + max_seq_len].to(self.device)
            attn = attn_mask[seq_ids, start : start + max_seq_len].to(self.device)

            if max_gen_len == 0:
                outputs = self.model(
                    input_ids=inputs,
                    attention_mask=attn,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states
            else:
                outputs = self.model.generate(
                    input_ids=inputs,
                    attention_mask=attn,
                    max_new_tokens=max_gen_len,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
                hidden_states = outputs.hidden_states[-1]

            self._accumulate_bi(hidden_states, angular=angular)

    def _accumulate_bi(self, hidden_states, angular: bool = False) -> None:
        self.bi_samples += 1
        for layer_idx in range(len(hidden_states) - 1):
            value = block_influence(
                hidden_states[layer_idx],
                hidden_states[layer_idx + 1],
                angular=angular,
            ).mean()
            self.bi_scores[layer_idx] += float(value.detach().cpu())

    @torch.inference_mode()
    def eval_q_similarity(
        self,
        prompts: List[str],
        max_seq_len: int,
        stride: int = 256,
        max_new_tokens: int = 1,
    ) -> None:
        prompt_tokens = self.tokenizer(
            prompts,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = prompt_tokens.input_ids
        attn_mask = prompt_tokens.attention_mask
        max_prompt_len = max(len(t) for t in input_ids)

        for start in range(0, max_prompt_len, stride):
            seq_ids = (attn_mask.sum(dim=-1) > start).nonzero().squeeze()
            seq_ids = seq_ids.unsqueeze(0) if seq_ids.dim() == 0 else seq_ids
            inputs = input_ids[seq_ids, start : start + max_seq_len].to(self.device)
            attn = attn_mask[seq_ids, start : start + max_seq_len].to(self.device)
            queries = self._collect_queries(inputs, attn, max_new_tokens=max_new_tokens)
            self._accumulate_q_similarity(queries)

    def _attention_query_shape(self, attn_module):
        num_heads = getattr(attn_module, "num_heads", None)
        if num_heads is None:
            num_heads = getattr(attn_module, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(self.model.config, "num_attention_heads", None)
        if num_heads is None:
            raise AttributeError("Cannot infer the number of query heads for this attention module")

        head_dim = getattr(attn_module, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                raise AttributeError("Cannot infer query head dimension for this attention module")
            head_dim = hidden_size // int(num_heads)
        return int(num_heads), int(head_dim)

    def _shape_query_states(self, q_states: torch.Tensor, attn_module) -> torch.Tensor:
        if q_states.dim() != 3:
            raise RuntimeError(f"Expected q_proj output with shape [batch, tokens, hidden], got {tuple(q_states.shape)}")
        batch, tokens, hidden = q_states.shape
        num_heads, head_dim = self._attention_query_shape(attn_module)
        expected = num_heads * head_dim
        if hidden != expected:
            if hidden % num_heads != 0:
                raise RuntimeError(f"Cannot reshape query output with hidden size {hidden} into {num_heads} heads")
            head_dim = hidden // num_heads
        return q_states.reshape(batch, tokens, num_heads, head_dim).transpose(1, 2).contiguous()

    def _collect_queries(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int):
        layer_outputs = {}

        def hook(attn_module, _inputs, output, layer_idx):
            q_states = self._shape_query_states(output.detach(), attn_module)
            layer_outputs.setdefault(layer_idx, []).append(q_states)

        hooks = []
        for layer_idx, layer in enumerate(self.layers):
            attn = getattr(layer, "self_attn", None)
            q_proj = getattr(attn, "q_proj", None)
            if q_proj is None:
                raise AttributeError(
                    "This model is not supported by the generic q-similarity collector because "
                    "its attention module does not expose q_proj."
                )
            hooks.append(q_proj.register_forward_hook(
                lambda module, inputs, output, layer_idx=layer_idx, attn=attn: hook(attn, inputs, output, layer_idx)
            ))
        try:
            if max_new_tokens > 0:
                self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            else:
                self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for handle in hooks:
                handle.remove()

        if len(layer_outputs) != len(self.layers):
            raise RuntimeError(f"Collected queries for {len(layer_outputs)} layers, expected {len(self.layers)}")

        per_layer = []
        for layer_idx in range(len(self.layers)):
            queries = torch.cat(layer_outputs[layer_idx], dim=2)
            queries = queries.squeeze(0) if queries.shape[0] == 1 else queries.mean(dim=0)
            if queries.dim() != 3:
                raise RuntimeError(f"Expected query tensor with shape [heads, tokens, dim], got {tuple(queries.shape)}")
            per_layer.append(queries)
        return torch.stack(per_layer, dim=0).to(torch.float32)

    def _accumulate_q_similarity(self, queries: torch.Tensor) -> None:
        self.q_samples += 1
        layer, head, length, dim = queries.shape
        flat = queries.reshape(layer * head, length, dim)
        norm = torch.norm(flat, dim=2, keepdim=True) + 1e-8
        flat = flat / norm
        cosine = torch.bmm(flat, flat.transpose(1, 2))
        q_mean = (cosine.sum(dim=(1, 2)) - length) / (length * (length - 1))
        per_layer = q_mean.reshape(layer, head).mean(dim=1)
        self.q_scores += per_layer.detach().cpu().numpy()

    def mean_bi(self) -> List[float]:
        if self.bi_samples == 0:
            raise RuntimeError("No BI samples have been accumulated")
        return (self.bi_scores / self.bi_samples).tolist()

    def mean_q_similarity(self) -> List[float]:
        if self.q_samples == 0:
            raise RuntimeError("No q-similarity samples have been accumulated")
        return (self.q_scores / self.q_samples).tolist()
