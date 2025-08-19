import torch


def unpad_input(
    inputs: "torch.Tensor",
    attention_mask: "torch.Tensor",
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]":
    """Remove padding from input sequences.

    Args:
        inputs (torch.Tensor, shape [batch, seqlen, ...]): tensor of token IDs.
        attention_mask (torch.Tensor, shape [batch, seqlen]): boolean token mask.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]

            - `unpadded_inputs` (torch.Tensor, shape [total_seq_len, ...]): the fused unpadded token IDs
            - `indices` (torch.Tensor, shape [total_seq_len, ...]): the sequence indices
            - `cu_seqlens` (torch.Tensor, [batch + 1,]): the cumulative sequence lengths
            - `max_seqlen_in_batch` (int): the maximum unpadded sequence length encountered in the batch

    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        shape = batch * seqlen
        unpadded_inputs = inputs.view(shape, *rest)[indices]

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch


def pad_output(
    inputs: "torch.Tensor",
    indices: "torch.Tensor",
    batch: int,
    seqlen: int,
) -> "torch.Tensor":
    """Add padding to sequences.

    Args:
        inputs (torch.Tensor, shape [total_nnz, ...]): Input tensor, unpadded.
        indices (torch.Tensor, shape [total_nnz,]): Indices tensor.
        batch (int): batch size
        seqlen (int): sequence length

    Returns:
        torch.Tensor
            The padded inputs, shape [batch, seqlen, ...]

    """
    if inputs.dim() == 1:
        output = torch.zeros(batch * seqlen, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.zeros(batch * seqlen, *rest, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen, *rest)

    return padded_inputs
