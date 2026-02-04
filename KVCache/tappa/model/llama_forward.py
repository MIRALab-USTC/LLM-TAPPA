import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.modeling_llama import *  # noqa: F403

from cake.cake_cache import CakeCache, CakeDecodingKVCache_LayerWise
from cake.utils import calculate_entropy

from tappa.tappa_qsim import compute_q_sim_torch


def llama_attn_forward_tappa(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    TAPPA-enabled FlashAttention2 forward for Llama.

    This function is intentionally kept close to CAKE's forward, with the TAPPA
    preference score variants implemented via `config.pref_score_type`.
    """
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    if isinstance(past_key_value, DynamicCache):
        past_key_value = CakeCache.from_dynamic_cache(past_key_value)

    if (
        self.config.decoding_evict[self.layer_idx] is None
        and len(past_key_value.layer_budget) == self.config.prefill_cake_evict[self.layer_idx].num_layers
    ):
        self.config.decoding_evict[self.layer_idx] = CakeDecodingKVCache_LayerWise(
            hh_size=past_key_value.layer_budget[self.layer_idx],
            window_size=self.config.window_size[self.layer_idx],
            k_seq_dim=2,
            v_seq_dim=2,
        )

    output_attentions = False
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(  # noqa: F405
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # noqa: F405

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)  # noqa: F405
    value_states = repeat_kv(value_states, self.num_key_value_groups)  # noqa: F405
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    if self.config.prefill[self.layer_idx]:
        tmp_attn_weights = torch.matmul(
            query_states[..., -self.config.window_size[self.layer_idx]:, :],
            key_states.transpose(2, 3),
        ) / math.sqrt(self.head_dim)

        if q_len != 1:
            mask = torch.full(
                (self.config.window_size[self.layer_idx], self.config.window_size[self.layer_idx]),
                torch.finfo(tmp_attn_weights.dtype).min,
                device=tmp_attn_weights.device,
            )
            mask_cond = torch.arange(mask.size(-1), device=tmp_attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            tmp_attention_mask = mask[None, None, :, :]
            tmp_attn_weights[:, :, -self.config.window_size[self.layer_idx]:, -self.config.window_size[self.layer_idx]:] += tmp_attention_mask

        tmp_attn_weights = nn.functional.softmax(tmp_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # TAPPA start
        T = self.config.window_size[self.layer_idx]
        q = query_states[0, :, -T:, :]
        mode = "cosine"
        q_sim = compute_q_sim_torch(q, mode=mode)

        if math.isinf(self.config.alpha):
            pref_score = 1 - q_sim
        else:
            disp = calculate_entropy(
                tmp_attn_weights[:, :, -T:, :-T]
            )
            var = torch.var(
                tmp_attn_weights[:, :, -T:, :-T],
                dim=-2,
            ).sum(0).sum(0).sum(0)

            cake_score = (disp ** (1 / self.config.tau1) * var ** (1 / self.config.tau2)).cpu().numpy()

            pref_score = cake_score + self.config.alpha * (1 - q_sim)
        # TAPPA end

        attention_score = tmp_attn_weights[..., -self.config.window_size[self.layer_idx]:, :]
        attn_mean = attention_score.mean(dim=-2)
        attn_var = attention_score.var(dim=-2)
        attn_cache = attn_mean + self.config.gamma * attn_var
        attn_cache = attn_cache[..., :-self.config.window_size[self.layer_idx]]
        attn_cache = F.avg_pool1d(attn_cache, kernel_size=5, padding=5 // 2, stride=1)

        attn_cache = attn_cache.reshape(bsz, self.num_key_value_heads, self.num_key_value_groups, -1)
        hh_score = attn_cache.mean(dim=-2)
        past_key_value.update_score(pref_score, hh_score)

    # In PEFT, layer norms or embeddings may be silently casted to float32.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(  # noqa: F405
            "The input hidden states seems to be silently casted in float32, this might be related to"
            " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
