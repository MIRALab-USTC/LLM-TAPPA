import torch


def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular: bool = False,
) -> torch.Tensor:
    """Compute ShortGPT block influence for a pair of hidden-state tensors."""
    _, _, hidden_size = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, hidden_size)
    output_hidden_state = output_hidden_state.reshape(-1, hidden_size)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)
    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return torch.arccos(sim) / torch.pi
    return 1 - sim
