from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import _mask_mod_signature

import torch


def doc_mask(attention_mask: "torch.Tensor") -> "_mask_mod_signature":
    """Create a document-level attention mask for multi-document sequences.

    This function creates a mask that prevents attention between tokens from
    different documents when multiple documents are packed into a single sequence.
    Each document is defined by a binary attention mask, and tokens can only
    attend to other tokens within the same document.

    Args:
        attention_mask: Binary attention mask of shape (n_batch, n_seq).

    Returns:
        _mask_mod_signature: A mask function compatible with flex_attention that
            returns True if query and key tokens are in the same document,
            False otherwise.

    """
    attention_mask = attention_mask.to(torch.bool)

    def __inner__(
        b: "torch.Tensor", _h: "torch.Tensor", q_idx: "torch.Tensor", kv_idx: "torch.Tensor"
    ) -> "torch.Tensor":
        """Inner mask function that checks if query and key indices are in the same document.

        Args:
            b: Batch index.
            _h: Head index (unused but required by signature).
            q_idx: Query token indices.
            kv_idx: Key/value token indices.

        Returns:
            torch.Tensor: Boolean mask where True indicates the query and key
                tokens are in the same document and can attend to each other.

        """
        return ~attention_mask[b, q_idx] & ~attention_mask[b, kv_idx]

    return __inner__
