from torch import nn
from torch.nn.modules.activation import (
    GELU,
    SELU,
    LeakyReLU,
    LogSigmoid,
    PReLU,
    ReLU,
    Sigmoid,
    SiLU,
)


def get_actv_fn(actv_fn: str) -> "nn.Module":
    """Get the activation function specified in the configuration.

    Args:
        actv_fn (str): Kind of activation function.

    Returns:
        nn.Module: An activation function module that can be called on tensors.

    Raises:
        ValueError: If the specified activation function is not supported.

    Supported activation functions:

        - `relu`: Rectified Linear Unit
        - `silu`: Sigmoid Linear Unit (Swish)
        - `gelu`: Gaussian Error Linear Unit
        - `leakyrelu`: Leaky Rectified Linear Unit
        - `selu`: Scaled Exponential Linear Unit
        - `logsigmoid`: Log-sigmoid activation
        - `sigmoid`: Standard sigmoid activation
        - `prelu`: Parametric Rectified Linear Unit

    """
    match actv_fn:
        case "relu":
            return ReLU()
        case "silu":
            return SiLU()
        case "gelu":
            return GELU()
        case "leakyrelu":
            return LeakyReLU()
        case "selu":
            return SELU()
        case "logsigmoid":
            return LogSigmoid()
        case "sigmoid":
            return Sigmoid()
        case "prelu":
            return PReLU()
        case _:
            supported_actv_fn = ["relu", "silu", "gelu", "leakyrelu", "selu", "logsigmoid", "sigmoid", "prelu"]
            raise ValueError(
                f"Unknown activation function {actv_fn}", f"Supported activation functions: {supported_actv_fn}"
            )


__all__ = ["get_actv_fn"]
