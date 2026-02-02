import torch


def unpad_input(
    inputs: "torch.Tensor",
    attention_mask: "torch.Tensor | None",
    pad_token_id: int | None = None,
    align_to: int = 1,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]":
    """Remove padding from input sequences.

    Args:
        inputs (torch.Tensor, shape [batch, seqlen, ...]): tensor of token IDs.
        attention_mask (torch.Tensor | None, shape [batch, seqlen]): boolean token mask, optional.
        pad_token_id (int | None): id of the padding token to remove, optional. Only used if attention_mask is None.
            If both are None, assumes full inputs.
        align_to (int): Round the total unpadded length up to the next multiple of this value by appending dummy
            padding tokens. This produces more stable tensor shapes across batches, improving ``torch.compile``
            compatibility and CUDA kernel throughput. The extra positions are masked out via ``cu_seqlens`` so they
            do not affect attention. Defaults to 1 (no alignment).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]

            - `unpadded_inputs` (torch.Tensor, shape [aligned_total_seq_len, ...]): the fused unpadded token IDs
            - `indices` (torch.Tensor, shape [total_seq_len, ...]): the sequence indices (without alignment padding)
            - `cu_seqlens` (torch.Tensor, [batch + 1,]): the cumulative sequence lengths
            - `max_seqlen_in_batch` (int): the maximum unpadded sequence length encountered in the batch
    """
    if attention_mask is None:
        if pad_token_id is not None:
            attention_mask = inputs != pad_token_id
        else:
            attention_mask = torch.ones(inputs.shape, device=inputs.device)

    seqlens_in_batch = attention_mask.sum(dim=-1)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0), (1, 0))

    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        shape = batch * seqlen
        unpadded_inputs = inputs.view(shape, *rest)[indices]

    if align_to > 1:
        total = unpadded_inputs.shape[0]
        aligned_total = (total + align_to - 1) // align_to * align_to
        if aligned_total > total:
            pad_size = aligned_total - total
            pad_value = pad_token_id if pad_token_id is not None else 0
            if unpadded_inputs.dim() == 1:
                padding = torch.full((pad_size,), pad_value, dtype=unpadded_inputs.dtype, device=unpadded_inputs.device)
            else:
                padding = torch.full(
                    (pad_size, *unpadded_inputs.shape[1:]),
                    pad_value,
                    dtype=unpadded_inputs.dtype,
                    device=unpadded_inputs.device,
                )
            unpadded_inputs = torch.cat([unpadded_inputs, padding])

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch


def pad_output(
    inputs: "torch.Tensor", indices: "torch.Tensor", batch: int, seqlen: int, pad_token_id: int | None = None
) -> "torch.Tensor":
    """Add padding to sequences.

    Args:
        inputs (torch.Tensor, shape [total_nnz, ...]): Input tensor, unpadded.
        indices (torch.Tensor, shape [total_nnz,]): Indices tensor.
        batch (int): batch size
        seqlen (int): sequence length
        pad_token_id (int): token ID to insert for padding.

    Returns:
        torch.Tensor
            The padded inputs, shape [batch, seqlen, ...]

    """
    pad_token_id = pad_token_id if pad_token_id is not None else 0
    if inputs.dim() == 1:
        output = torch.full((batch * seqlen,), pad_token_id, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.full((batch * seqlen, *rest), pad_token_id, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen, *rest)

    return padded_inputs


__all__ = ["pad_output", "unpad_input"]
