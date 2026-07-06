from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class BagOfWordsLoss(nn.Module):
    """Bag-of-subword modeling loss on a mean-pooled sequence representation.

    Given per-document bag-of-word logits (produced by projecting the mean-pooled sequence
    embedding to the vocabulary), this computes a binary cross-entropy against the *multi-hot* set
    of subwords present in the unmasked sequence: the target is ``1`` for every vocabulary id that
    occurs anywhere in the document and ``0`` otherwise. Structural tokens (padding, ``[CLS]``,
    ``[SEP]``, ...) are excluded from the target so the head is not rewarded for predicting them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    @staticmethod
    def build_multihot(
        token_ids: "torch.Tensor",
        segment_ids: "torch.Tensor",
        num_segments: int,
        vocab_size: int,
        ignore_token_ids: "list[int] | None" = None,
    ) -> "torch.Tensor":
        """Build a multi-hot ``[num_segments, vocab_size]`` presence target.

        Args:
            token_ids (torch.Tensor, shape [num_tokens]): Vocabulary id of each valid token (in the
                unmasked/original sequence), in segment order.
            segment_ids (torch.Tensor, shape [num_tokens]): Document index of each token.
            num_segments (int): Number of documents.
            vocab_size (int): Vocabulary size (target dimensionality).
            ignore_token_ids (list[int], optional): Vocabulary ids to zero out in the target (special
                tokens). Defaults to None.

        Returns:
            torch.Tensor, shape [num_segments, vocab_size]: Float multi-hot presence target.
        """
        target = torch.zeros(num_segments, vocab_size, dtype=torch.float, device=token_ids.device)
        target[segment_ids, token_ids] = 1.0
        if ignore_token_ids:
            ignore = torch.tensor(ignore_token_ids, device=token_ids.device)
            ignore = ignore[(ignore >= 0) & (ignore < vocab_size)]
            target[:, ignore] = 0.0
        return target

    def forward(self, bow_logits: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        """Compute the bag-of-words BCE loss.

        Args:
            bow_logits (torch.Tensor, shape [num_segments, vocab_size]): Predicted per-document logits.
            target (torch.Tensor, shape [num_segments, vocab_size]): Multi-hot presence target.

        Returns:
            torch.Tensor: Scalar loss.
        """
        loss: torch.Tensor = self.loss_fn(bow_logits, target)
        return loss


class InBatchSimilarityLoss(nn.Module):
    """Isotropy loss driving the batch's mean-pooled sequence embeddings toward mutual orthogonality.

    Penalizes the pairwise cosine similarity between all sequence embeddings in a batch toward 0 by
    minimizing the mean of the squared off-diagonal entries of the cosine-similarity matrix. With
    fewer than two sequences there are no pairs and the loss is 0.

    Args:
        gather_distributed (bool): If True and running under an initialized process group, all-gather
            the pooled embeddings across ranks so the similarity is computed over the global batch.
            Defaults to False.
    """

    def __init__(self, gather_distributed: bool = False) -> None:
        super().__init__()
        self.gather_distributed = gather_distributed

    def forward(self, pooled: "torch.Tensor") -> "torch.Tensor":
        """Compute the in-batch similarity (isotropy) loss.

        Args:
            pooled (torch.Tensor, shape [num_sequences, hidden]): Mean-pooled sequence embeddings.

        Returns:
            torch.Tensor: Scalar loss (mean squared off-diagonal cosine similarity).
        """
        if self.gather_distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [torch.zeros_like(pooled) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered, pooled.contiguous())
            gathered[torch.distributed.get_rank()] = pooled  # keep local grad path
            pooled = torch.cat(gathered, dim=0)

        num = pooled.shape[0]
        if num < 2:
            return pooled.sum() * 0.0
        normed = F.normalize(pooled, dim=-1)
        similarity = normed @ normed.t()
        squared = similarity.pow(2)
        off_diagonal = squared.sum() - squared.diagonal().sum()
        return off_diagonal / (num * (num - 1))


def get_loss_function(
    problem_type: Literal["regression", "single_label_classification", "multi_label_classification"] | None,
) -> "nn.Module":
    """Return the applicable loss function for a given problem type.

    Args:
        problem_type (Literal["regression", "single_label_classification", "multi_label_classification"] | None):
            The type of problem.

    Returns:
        nn.Module: The appropriate loss function module.

    Raises:
        ValueError: If the problem type is not supported.

    """
    if problem_type == "regression":
        return nn.MSELoss()
    elif problem_type == "single_label_classification":
        return nn.CrossEntropyLoss()
    elif problem_type == "multi_label_classification":
        return nn.BCEWithLogitsLoss()
    else:
        supported_types = ["regression", "single_label_classification", "multi_label_classification"]
        raise ValueError(f"Unknown problem type '{problem_type}'. Supported types: {', '.join(supported_types)}")


__all__ = ["BagOfWordsLoss", "InBatchSimilarityLoss", "get_loss_function"]
