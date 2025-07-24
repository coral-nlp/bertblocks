from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import _mask_mod_signature

import torch


def doc_mask(attention_mask: "torch.Tensor") -> "_mask_mod_signature":
    """Create a document-level attention mask to prevent cross-document attention.

    Transforms a given binary attention mask into a block mask compatible with flex-attention.

    Args:
        attention_mask (torch.Tensor, shape [n_batch, n_seq]): Binary attention mask.

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
            b (torch.Tensor): Batch index.
            _h (torch.Tensor): Head index (unused but required by signature).
            q_idx (torch.Tensor): Query token indices.
            kv_idx (torch.Tensor): Key/value token indices.

        Returns:
            torch.Tensor: Boolean mask where True indicates the query and key
                tokens are in the same document and can attend to each other.

        """
        return attention_mask[b, q_idx] & attention_mask[b, kv_idx]

    return __inner__
