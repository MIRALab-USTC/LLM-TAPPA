import transformers

from cake.model.modify_llama import llama_model_forward_cake
from cake.model.modify_qwen2 import qwen2_model_forward_cake

from tappa.model.llama_forward import llama_attn_forward_tappa
from tappa.model.qwen2_forward import qwen2_attn_forward_tappa


def patch_llama() -> None:
    """
    Patch HuggingFace Llama FlashAttention2 to use TAPPA attention forward.
    """
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_cake
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_attn_forward_tappa


def patch_qwen2() -> None:
    """
    Patch HuggingFace Qwen2 FlashAttention2 to use TAPPA attention forward.
    """
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen2_model_forward_cake
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_attn_forward_tappa


def apply_tappa_patch(model_type: str) -> None:
    """
    Apply TAPPA patch by model type.

    Parameters
    ----------
    model_type:
        "llama" or "qwen2"
    """
    mt = model_type.lower()
    if mt in ("llama", "llama3", "llama3.1"):
        patch_llama()
    elif mt in ("qwen2", "qwen2.5"):
        patch_qwen2()
    else:
        raise ValueError(f"Unsupported model_type for TAPPA patch: {model_type}")
