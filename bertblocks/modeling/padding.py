import torch


def _unpad_packed_input(
    inputs: "torch.Tensor",
    attention_mask: "torch.Tensor",
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]":
    """Remove padding from packed input sequences.

    Handles packed sequence format where attention_mask contains sequence indices
    (0, 1, 2, ...) with -1 marking padding tokens.
    """
    # Flatten to process all sequences at once
    flat_mask = attention_mask.view(-1)

    # Handle both [B, S] and [B, S, H] inputs
    if inputs.dim() > 2:
        B, S, H = inputs.shape
        flat_inputs = inputs.reshape(B * S, H).contiguous()
    else:
        flat_inputs = inputs.reshape(-1).contiguous()

    # Extract valid tokens (mask >= 0)
    valid_mask = flat_mask >= 0
    indices = torch.nonzero(valid_mask, as_tuple=False).flatten()

    # Create a new contiguous tensor, which is safer for Flash Attention
    unpadded_inputs = flat_inputs[indices]
    valid_seq_indices = flat_mask[indices]

    if len(valid_seq_indices) == 0:
        return unpadded_inputs, indices, torch.tensor([0], device=inputs.device), 0

    # Count occurrences of each sequence ID using bincount
    seqlens = torch.bincount(valid_seq_indices)
    max_seqlen_in_batch = int(seqlens.max().item())
    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=inputs.device)
    torch.cumsum(seqlens, dim=0, out=cu_seqlens[1:])
    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch


def _unpad_standard_input(
    input_ids: "torch.Tensor", attention_mask: "torch.Tensor"
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]":
    """Remove padding from standard (non-packed) input sequences."""
    seqlens_in_batch = attention_mask.sum(dim=-1)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0), (1, 0))

    if input_ids.dim() == 2:
        unpadded_inputs = input_ids.flatten()[indices]
    else:
        batch, seqlen, *rest = input_ids.shape
        shape = batch * seqlen
        unpadded_inputs = input_ids.view(shape, *rest)[indices]

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch


def unpad_input(
    input_ids: "torch.Tensor",
    attention_mask: "torch.Tensor | None",
    pad_token_id: int | None = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]":
    """Remove padding from input sequences.

    Automatically detects and handles both standard (binary 0/1) and packed (sequence-indexed)
    attention mask formats.

    Args:
        input_ids (torch.Tensor, shape [batch, seqlen, ...]): tensor of token IDs.
        attention_mask (torch.Tensor | None, shape [batch, seqlen]): token mask.
            Can be binary (standard) or sequence-indexed (packed).
        pad_token_id (int | None): id of the padding token to remove, optional. Only used if attention_mask is None.
            If both are None, assumes full inputs.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]

            - `unpadded_inputs` (torch.Tensor, shape [total_seq_len, ...]): the fused unpadded token IDs
            - `indices` (torch.Tensor, shape [total_seq_len, ...]): the sequence indices
            - `cu_seqlens` (torch.Tensor, [batch + 1,]): the cumulative sequence lengths
            - `max_seqlen_in_batch` (int): the maximum unpadded sequence length encountered in the batch
    """
    # Detect and route to packed format handler if needed
    if attention_mask is not None and ((attention_mask > 1).any() or (attention_mask < 0).any()):
        return _unpad_packed_input(input_ids, attention_mask)
    else:
        if attention_mask is None:
            if pad_token_id is not None:
                attention_mask = input_ids != pad_token_id
            else:
                attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
        return _unpad_standard_input(attention_mask, input_ids)


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
