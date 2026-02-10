"""Initialization functions; includes both from-scratch initalization and tiling initialization from another model.

References:
    - [ModernBERT Initialization Strategies](https://github.com/AnswerDotAI/ModernBERT/blob/8c57a0f01c12c4953ead53d398a36f81a4ba9e38/src/bert_layers/initialization.py#L68)
"""

import math
from enum import StrEnum  # type: ignore
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from typing_extensions import Self

if TYPE_CHECKING:
    from bertblocks.config import BertBlocksConfig


__all__ = ["TilableMixin", "TileLinear", "TileMode", "InitFnType", "init_weights", "get_layer_dim"]


class InitFnType(StrEnum):
    """Available weight initialization functions."""

    normal = "normal"
    """
    All weights are initialized from the same normal distribution.
    """

    trunc_normal = "trunc_normal"
    """
    All weights are initialized from a truncated normal distribution.
    Equivalent to 'normal' when initializer_cutoff_factor is set.
    """

    kaiming_normal = "kaiming_normal"
    """
    All weights are initialized with the Kaiming method from a normal distribution.
    Note this currently won't work with FSDP.
    """

    kaiming_uniform = "kaiming_uniform"
    """
    All weights are initialized with the Kaiming method from a uniform distribution.
    Note this currently won't work with FSDP.
    """

    xavier_normal = "xavier_normal"
    """
    All weights are initialized with the Xavier/Glorot method from a normal distribution.
    """

    xavier_uniform = "xavier_uniform"
    """
    All weights are initialized with the Xavier/Glorot method from a uniform distribution.
    """

    fan_in = "fan_in"
    """
    Fan-in variance scaling: normal distribution with std = ``1/sqrt(d_in)`` where ``d_in``
    is the input dimensionality of the kernel.
    """


def get_layer_dim(module: "nn.Module") -> int | None:
    """Get effective input dimensionality for a module.

    Args:
        module: Linear or Embedding module

    Returns:
        Input dimension or None
    """
    if isinstance(module, nn.Linear):
        return int(module.in_features)
    elif isinstance(module, nn.Embedding):
        return int(module.embedding_dim)
    return None


def init_weights(
    config: "BertBlocksConfig",
    module: nn.Linear | nn.Embedding,
    layer_dim: int | None = None,
    std_factor: float = 1.0,
) -> None:
    """Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param layer_dim: The effective input dimensionality of the weights. This could be smaller than the actual dims.
        for fused layers.
    :param std_factor: Multiplier for the standard deviation.
    """
    layer_dim = layer_dim if layer_dim is not None else config.hidden_size

    if config.initializer_kind in (InitFnType.normal, InitFnType.trunc_normal):
        std = config.initializer_range * std_factor
        if config.initializer_cutoff_factor is not None:
            cutoff_value = config.initializer_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.initializer_kind == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.initializer_kind == InitFnType.kaiming_uniform:
        nn.init.kaiming_uniform_(module.weight)
    elif config.initializer_kind == InitFnType.xavier_normal:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
    elif config.initializer_kind == InitFnType.xavier_uniform:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
    elif config.initializer_kind == InitFnType.fan_in:
        std = std_factor / math.sqrt(layer_dim)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    else:
        raise NotImplementedError(config.initializer_kind)

    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)

    if isinstance(module, nn.Embedding) and config.init_small_embedding:
        nn.init.uniform_(module.weight, a=-1e-4, b=1e-4)


class TileMode(StrEnum):
    """Available tiling strategies."""

    center_weights = "center_weights"
    tile_weights_from_edge = "tile_weights_from_edge"
    tile_weights_from_middle = "tile_weights_from_middle"


class TileLinear(StrEnum):
    """Special cases for tiling handling."""

    wqkv = "wqkv"
    glu = "glu"
    default = "default"


class TilableMixin:
    """Mixin for modules that have special weight-tiling logic.

    Modules with fused projections (e.g., fused QKV in attention, fused gate+input in GLU)
    should inherit from this mixin and implement ``tile_from`` to define how their weights
    should be tiled when expanding from a smaller pretrained model.

    The ``_init_with_tiling`` method in ``BertBlocksPreTrainedModel`` iterates over all modules
    and calls ``tile_from`` for any ``TilableMixin`` instances, then handles remaining plain
    modules (``nn.Linear``, ``nn.Embedding``, normalization layers) generically. Children of
    a ``TilableMixin`` module are skipped to avoid double-tiling.

    Example::

        class Attention(TilableMixin, nn.Module):
            def tile_from(self, pretrained: "Attention", mode: TileMode) -> None:
                tile_linear(pretrained.proj, self.proj, linear_type=TileLinear.wqkv, mode=mode)
                tile_linear(pretrained.ffwd, self.ffwd, mode=mode)
    """

    def tile_from(
        self,
        pretrained: Self,
        mode: str | TileMode = TileMode.tile_weights_from_middle,
    ) -> None:
        """Tile weights from a smaller pretrained module into self.

        Args:
            pretrained: The pretrained module with smaller dimensions to tile from.
            mode: Tiling strategy to use.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement tile_from")


def tile_weight(
    pretrained_weights: torch.Tensor,
    new_weights: torch.Tensor,
    mode: str | TileMode = TileMode.tile_weights_from_middle,
) -> torch.Tensor:
    """Tile or center an input tensor to a larger desired size. Works for both 2D and 1D tensors.

    Args:
    pretrained_weights (torch.Tensor): The input tensor to be tiled or centered (1D or 2D).
    new_weights (torch.Tensor): The tensor with the desired size.
    mode (Union[str, TileMode]): 'center_weights', 'tile_weights_from_edge', or 'tile_weights_from_middle'

    Returns:
    torch.Tensor: The resulting tensor of the desired size.
    """
    assert pretrained_weights.dim() in (1, 2), "Input tensor must be 1-dimensional or 2-dimensional"
    if isinstance(mode, str):
        mode = TileMode(mode)

    pretrained_weights = pretrained_weights.clone()

    if pretrained_weights.dim() == 1:
        return _tile_1d(pretrained_weights, new_weights, mode)
    else:
        return _tile_2d(pretrained_weights, new_weights, mode)


def _tile_1d(pretrained_weights: torch.Tensor, new_weights: torch.Tensor, mode: TileMode) -> torch.Tensor:
    assert pretrained_weights.dim() == 1, "Input tensor must be 1-dimensional"
    input_size = pretrained_weights.shape[0]
    new_size = new_weights.shape[0]
    assert new_size >= input_size, "Desired size must be greater than or equal to input size"

    if mode == TileMode.center_weights:
        offset = (new_size - input_size) // 2
        new_weights[offset : offset + input_size] = pretrained_weights
        return new_weights.clone()
    elif mode == TileMode.tile_weights_from_edge:
        repeat_count = (new_size + input_size - 1) // input_size
        tiled_tensor = pretrained_weights.repeat(repeat_count)
        return tiled_tensor[:new_size].clone()
    elif mode == TileMode.tile_weights_from_middle:
        # Calculate offsets to center the original tensor
        offset = (new_size - input_size) // 2

        # Create a new tensor with the desired size
        result = torch.zeros(new_size, dtype=pretrained_weights.dtype, device=pretrained_weights.device)

        # Place the original tensor in the center
        result[offset : offset + input_size] = pretrained_weights

        # Tile the left and right sides
        for i in range(offset):
            result[offset - 1 - i] = pretrained_weights[input_size - 1 - (i % input_size)]
        for i in range(offset + input_size, new_size):
            result[i] = pretrained_weights[(i - offset) % input_size]
        return result.clone()
    else:
        raise ValueError(f"Unknown tile mode: {mode}")


def _tile_2d(pretrained_weights: torch.Tensor, new_weights: torch.Tensor, mode: TileMode) -> torch.Tensor:
    assert pretrained_weights.dim() == 2, "Input tensor must be 2-dimensional"
    input_height, input_width = pretrained_weights.shape
    new_height, new_width = new_weights.shape
    assert new_height >= input_height, "Desired height must be greater than or equal to input height"
    assert new_width >= input_width, "Desired width must be greater than or equal to input width"

    if mode == TileMode.center_weights:
        height_offset = (new_height - input_height) // 2
        width_offset = (new_width - input_width) // 2
        new_weights[height_offset : height_offset + input_height, width_offset : width_offset + input_width] = (
            pretrained_weights
        )
        return new_weights.clone()
    elif mode == TileMode.tile_weights_from_edge:
        repeat_height = (new_height + input_height - 1) // input_height
        repeat_width = (new_width + input_width - 1) // input_width
        tiled_tensor = pretrained_weights.repeat(repeat_height, repeat_width)
        return tiled_tensor[:new_height, :new_width].clone()
    elif mode == TileMode.tile_weights_from_middle:
        # Calculate offsets to center the original tensor
        height_offset = (new_height - input_height) // 2
        width_offset = (new_width - input_width) // 2

        # Create a new tensor with the desired width and input height
        horizontal_tiled = torch.zeros(
            input_height, new_width, dtype=pretrained_weights.dtype, device=pretrained_weights.device
        )

        # Place the original tensor in the center horizontally
        horizontal_tiled[:, width_offset : width_offset + input_width] = pretrained_weights

        # Tile the left and right sides
        for i in range(width_offset):
            horizontal_tiled[:, i] = horizontal_tiled[
                :, width_offset + input_width - 1 - (width_offset - i - 1) % input_width
            ]
        for i in range(width_offset + input_width, new_width):
            horizontal_tiled[:, i] = horizontal_tiled[:, width_offset + (i - width_offset) % input_width]

        # Now tile vertically
        result = torch.zeros(new_height, new_width, dtype=pretrained_weights.dtype, device=pretrained_weights.device)
        result[height_offset : height_offset + input_height, :] = horizontal_tiled

        # Tile top
        for i in range(height_offset):
            row_to_copy = (input_height - 1) - (i % input_height)
            result[height_offset - 1 - i, :] = horizontal_tiled[row_to_copy, :]

        # Tile bottom
        for i in range(height_offset + input_height, new_height):
            row_to_copy = (i - height_offset) % input_height
            result[i, :] = horizontal_tiled[row_to_copy, :]
        return result.clone()
    else:
        raise ValueError(f"Unknown tile mode: {mode}")


def tile_fused_qkv(
    pretrained_qkv_weight: torch.Tensor,
    new_qkv_weight: torch.Tensor,
    mode: str | TileMode = TileMode.tile_weights_from_middle,
) -> torch.Tensor:
    """Tile the weights of a fused pretrained QKV layer to a new, larger QKV dimension.

    Args:
        pretrained_qkv_weight (torch.Tensor): The original fused QKV layer
        new_qkv_weight (torch.Tensor): The new fused QKV layer with larger linear_dim
        mode (Union[str, TileMode]): The tiling mode to use
    Returns:
        torch.Tensor: The new fused QKV layer with tiled weights
    """
    # Split QKV, assume new_q, new_k, new_v are the same shape
    pretrained_q, pretrained_k, pretrained_v = pretrained_qkv_weight.chunk(3, dim=0)
    new_q, new_k, new_v = new_qkv_weight.chunk(3, dim=0)

    # Tile Q, K, V separately
    new_q = tile_weight(pretrained_q, new_q, mode=mode)
    new_k = tile_weight(pretrained_k, new_k, mode=mode)
    new_v = tile_weight(pretrained_v, new_v, mode=mode)

    # Concatenate tiled Q, K, V
    return torch.cat([new_q, new_k, new_v], dim=0)


def tile_fused_glu(
    pretrained_glu_weight: torch.Tensor,
    new_glu_weight: torch.Tensor,
    mode: str | TileMode = TileMode.tile_weights_from_middle,
) -> torch.Tensor:
    """Tile the weights of a fused pretrained GLU layer to a new, larger GLU dimension.

    Args:
        pretrained_glu_weight (torch.Tensor): The original fused GLU layer
        new_glu_weight (torch.Tensor): The new fused GLU layer with larger linear_dim
        mode (Union[str, TileMode]): The tiling mode to use
    Returns:
        torch.Tensor: The new fused GLU layer with tiled weights
    """
    # Split GLU, assume new_glu_wi, new_glu_wg are the same shape
    pretrained_glu_wi, pretrained_glu_wg = pretrained_glu_weight.chunk(2, dim=0)
    new_glu_wi, new_glu_wg = new_glu_weight.chunk(2, dim=0)

    # Tile GLU separately
    new_glu_wi = tile_weight(pretrained_glu_wi, new_glu_wi, mode=mode)
    new_glu_wg = tile_weight(pretrained_glu_wg, new_glu_wg, mode=mode)

    # Concatenate tiled GLU
    return torch.cat([new_glu_wi, new_glu_wg], dim=0)


def tile_linear(
    pretrained_linear: nn.Linear,
    new_linear: nn.Linear,
    linear_type: str | TileLinear = TileLinear.default,
    mode: str | TileMode = TileMode.tile_weights_from_middle,
    bias_only: bool | None = False,
) -> None:
    """Tile the weights of a linear layer to a new, larger linear dimension.

    Args:
        pretrained_linear (nn.Linear): The original linear layer
        new_linear (nn.Linear): The new linear layer with larger linear_dim
        linear_type (Union[str, TileLinear]): The type of linear layer to tile
        mode (Union[str, TileMode]): The tiling mode to use
        bias_only (bool): Whether to only tile the bias. Only used if tiling weight tied decoder.
    """
    if isinstance(linear_type, str):
        linear_type = TileLinear(linear_type)
    if isinstance(mode, str):
        mode = TileMode(mode)

    with torch.no_grad():
        if linear_type == TileLinear.wqkv:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    tile_fused_qkv(pretrained_linear.weight, new_linear.weight, mode=mode),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    tile_fused_qkv(pretrained_linear.bias, new_linear.bias, mode=mode),
                    requires_grad=new_linear.bias.requires_grad,
                )
        elif linear_type == TileLinear.glu:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    tile_fused_glu(pretrained_linear.weight, new_linear.weight, mode=mode),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    tile_fused_glu(pretrained_linear.bias, new_linear.bias, mode=mode),
                    requires_grad=new_linear.bias.requires_grad,
                )
        else:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    tile_weight(pretrained_linear.weight, new_linear.weight, mode=mode),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    tile_weight(pretrained_linear.bias, new_linear.bias, mode=mode),
                    requires_grad=new_linear.bias.requires_grad,
                )


def tile_norm(
    pretrained_norm: "nn.Module",
    new_norm: "nn.Module",
    mode: str | TileMode = TileMode.tile_weights_from_middle,
) -> None:
    """Tile the weights of a pretrained norm layer to a new, larger layer norm dimension.

    Args:
        pretrained_norm (Union[nn.LayerNorm, RMSNorm, nn.Identity]): The original norm layer
        new_norm (Union[nn.LayerNorm, RMSNorm, nn.Identity]): The new norm layer with larger layer norm dimension
        mode (Union[str, TileMode]): The tiling mode to use
    """
    if isinstance(pretrained_norm, nn.Identity):
        return
    if isinstance(mode, str):
        mode = TileMode(mode)

    with torch.no_grad():
        new_norm.weight.data = nn.Parameter(
            tile_weight(pretrained_norm.weight, new_norm.weight, mode=mode),
            requires_grad=new_norm.weight.requires_grad,
        )
        if hasattr(pretrained_norm, "bias") and pretrained_norm.bias is not None:
            new_norm.bias.data = nn.Parameter(
                tile_weight(pretrained_norm.bias, new_norm.bias, mode=mode),
                requires_grad=new_norm.bias.requires_grad,
            )


def tile_embedding(
    pretrained_embedding: nn.Embedding,
    new_embedding: nn.Embedding,
    mode: str | TileMode = TileMode.tile_weights_from_middle,
) -> None:
    """Tile the weights of an embedding layer to a new, larger embedding dimension.

    Args:
    pretrained_embedding (nn.Embedding): The original embedding layer
    new_embedding (nn.Embedding): The new embedding layer with larger embedding_dim
    mode (Union[str, TileMode]): The tiling mode to use

    Returns:
    nn.Embedding: The new embedding layer with tiled weights
    """
    with torch.no_grad():
        # Ensure vocabulary size remains the same
        if pretrained_embedding.num_embeddings != new_embedding.num_embeddings:
            raise ValueError("Vocabulary size (num_embeddings) must remain constant")

        # Ensure new embedding dimension is larger
        if new_embedding.embedding_dim <= pretrained_embedding.embedding_dim:
            raise ValueError("New embedding_dim must be larger than the old embedding_dim")

        # Tile the weights
        new_embedding.weight.data = nn.Parameter(
            tile_weight(pretrained_embedding.weight, new_embedding.weight, mode=mode),
            requires_grad=new_embedding.weight.requires_grad,
        )

        # Handle padding_idx if it exists
        if pretrained_embedding.padding_idx is not None:
            if new_embedding.padding_idx is None:
                new_embedding.padding_idx = pretrained_embedding.padding_idx
            else:
                assert new_embedding.padding_idx == pretrained_embedding.padding_idx, "padding_idx must remain the same"
