"""Tensor padding and unpadding utilities for efficient attention computation.

This module provides utilities for padding and unpadding variable-length sequences
into dense tensors, enabling efficient computation with unpadded sequences.

Partly adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.attention.flex_attention import _mask_mod_signature


class IndexFirstAxis(torch.autograd.Function):
    """Custom autograd function for indexing along the first axis of a tensor.

    This function efficiently selects elements from the first dimension of a tensor
    using provided indices, maintaining gradient flow. It's particularly useful for
    extracting non-padded tokens from a batch of sequences.
    """

    @staticmethod
    def forward(ctx, input, indices):  # type: ignore
        """Forward pass: select elements from input using indices.

        Args:
            ctx: Autograd context for saving information for backward pass.
            input: Input tensor of shape (batch_size, ...) to index from.
            indices: 1D tensor of indices to select from the first dimension.

        Returns:
            torch.Tensor: Selected elements reshaped to (-1, *other_dimensions).

        """
        ctx.save_for_backward(indices)
        assert input.ndim >= 2  # nosec B101
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)).reshape(
            -1, *other_shape
        )

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        """Backward pass: scatter gradients back to original positions.

        Args:
            ctx: Autograd context containing saved indices.
            grad_output: Gradients from the subsequent computation.

        Returns:
            tuple[torch.Tensor, None]: Gradients for input (scattered back to
                original shape) and None for indices.

        """
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2  # nosec B101
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    """Custom autograd function for placing values at specified indices along the first axis.

    This function creates a new tensor and places the input values at the specified
    indices along the first dimension. It's the inverse operation of IndexFirstAxis
    and is used for unpacking sequences back to their original padded format.
    """

    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):  # type: ignore
        """Forward pass: place values at specified indices in a new tensor.

        Args:
            ctx: Autograd context for saving information for backward pass.
            values: Input tensor of shape (num_indices, ...) containing values to place.
            indices: 1D tensor of indices where to place the values.
            first_axis_dim: Size of the first dimension for the output tensor.

        Returns:
            torch.Tensor: Output tensor of shape (first_axis_dim, ...) with values
                placed at the specified indices, zeros elsewhere.

        """
        ctx.save_for_backward(indices)
        assert indices.ndim == 1  # nosec B101
        assert values.ndim >= 2  # nosec B101
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output.scatter_(0, repeat(indices, "z -> z d", d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        """Backward pass: gather gradients from the specified indices.

        Args:
            ctx: Autograd context containing saved indices.
            grad_output: Gradients from the subsequent computation.

        Returns:
            tuple[torch.Tensor, None, None]: Gradients for values (gathered from
                specified indices) and None for indices and first_axis_dim.

        """
        (indices,) = ctx.saved_tensors
        grad_values = torch.gather(grad_output, 0, repeat(indices, "z -> z d", d=grad_output.shape[1]))
        return grad_values, None, None


# Create convenient function handles for the custom autograd functions
index_put_first_axis = IndexPutFirstAxis.apply


def unpad_sequence(
    hidden_states: "torch.Tensor", attention_mask: "torch.Tensor | None"
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Pack variable-length sequences into a dense tensor by removing padding.

    This function takes batched sequences with padding and unpads them into a single
    dense tensor containing only the valid (non-padded) tokens.

    Args:
        hidden_states (Tensor, shape [batch_size, sequence_length, ...]): Input tensor of token representations.
        attention_mask (Tensor, shape [batch_size, sequence_length]): Optional mask tensor where
            1 indicates valid tokens and 0 indicates padding. If None, all tokens are considered valid.

    Returns:
        tuple containing:
        - unpadded_sequence: Tensor of shape (total_valid_tokens, ...) containing
            only the valid tokens from all sequences.
        - indices: 1D tensor containing the flattened indices of valid tokens.
        - cu_seqlens: Cumulative sequence lengths tensor of shape (batch_size + 1)
            used for indexing into the packed tensor.
        - max_seqlen_in_batch: Maximum sequence length in the batch.

    Example:
        >>> hidden_states = torch.randn(2, 4, 768)  # 2 sequences, max_len=4
        >>> attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])  # lengths [3, 2]
        >>> packed, indices, cu_seqlens, max_len = unpad_sequence(hidden_states, attention_mask)
        >>> packed.shape  # torch.Size([5, 768]) - only 5 valid tokens total
        >>> cu_seqlens  # tensor([0, 3, 5]) - cumulative lengths

    """
    if attention_mask is None:
        attention_mask = torch.ones_like(hidden_states)
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_sequence(
    hidden_states: "torch.FloatTensor", indices: "torch.Tensor", batch: int, seqlen: int
) -> "torch.FloatTensor":
    """Pad a dense tensor back into batched sequences with padding.

    This function reverses the unpad operation, taking a dense tensor of valid
    tokens and reconstructing the original batched format with appropriate padding.
    Padded positions will contain zeros.

    Args:
        hidden_states (Tensor, shape [num_tokens, ...]): Dense tensor of token representations.
        indices (Tensor, shape [num_tokens]): 1D tensor containing the flattened
            indices where each token should be placed in the unpacked tensor.
        batch (int): Batch size for the output tensor.
        seqlen (int): Maximum sequence length for the output tensor.

    Returns:
        torch.FloatTensor: Unpacked tensor of shape (batch, seqlen, ...) with
            valid tokens placed at their original positions and zeros for padding.

    Example:
        >>> # Following the pack_seq example
        >>> padded = pad_sequence(packed, indices, batch=2, seqlen=4)
        >>> padded.shape  # torch.Size([2, 4, 768]) - back to original shape

    """
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def get_block_mask_mod(cu_seqlens: "torch.Tensor") -> "_mask_mod_signature":
    """Create a document-level attention mask for multi-document sequences.

    This function creates a mask that prevents attention between tokens from
    different documents when multiple documents are packed into a single sequence.
    Each document is defined by cumulative sequence lengths, and tokens can only
    attend to other tokens within the same document.

    Args:
        cu_seqlens: Cumulative sequence lengths tensor of shape (num_docs + 1)
            where cu_seqlens[i] represents the starting position of document i.
            The last element should be the total sequence length.

    Returns:
        _mask_mod_signature: A mask function compatible with flex_attention that
            returns True if query and key tokens are in the same document,
            False otherwise.

    """

    def __inner__(
        _b: "torch.Tensor", _h: "torch.Tensor", q_idx: "torch.Tensor", kv_idx: torch.Tensor
    ) -> "torch.Tensor":
        """Inner mask function that checks if query and key indices are in the same document."""
        # Compare pseudo-docids (number of cumulative lengths exceeded)
        q_doc_id = (q_idx >= cu_seqlens).sum(-1)
        kv_doc_id = (kv_idx >= cu_seqlens).sum(-1)
        return q_doc_id == kv_doc_id

    return __inner__
