import warnings
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from torch.optim import Optimizer


def get_optimizer(
    optimizer_name: Literal[
        "adafactor",
        "adagrad",
        "adam",
        "adamw",
        "galore",
        "lamb",
        "lars",
        "lion",
        "muon",
        "sgd",
        "shampoo",
        "soap",
        "sophiah",
        "splus",
        "rmsprop",
    ],
    params: list[dict[str, Any]],
    optimizer_kwargs: dict[str, Any],
    quantized: bool = False,
) -> "Optimizer":
    """Instantiate a specific optimizer with params and hyperparameters and return it.

    Args:
        optimizer_name (str): Name of the optimizer class to instantiate.
        params (list[dict[str, Any]]): Model parameters to be optimized.
        optimizer_kwargs (dict[str, Any]): Optional hyperparameters to pass to optimizer.
        quantized (bool): Whether to use a 8bit-quantized optimizer variant.

    Returns:
        The instantiated optimizer.

    Raises:
        ValueError: If the specified optimizer name is not recognized.
        ImportError: If the specified optimizer name requires a non-installed optional dependency.

    Supported optimizer types:
        - `adafactor`
        - `adagrad`
        - `adam`
        - `adamw`
        - `galore`
        - `lamb`
        - `lars`
        - `lion`
        - `muon`
        - `sgd`
        - `shampoo`
        - `soap`
        - `sophiah`
        - `splus`
        - `rmsprop`

    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    match optimizer_name:
        case "adafactor":
            if quantized:
                warnings.warn(
                    f"No quantized option available for {optimizer_name}, falling back to unquantized optimizer",
                    stacklevel=1,
                )
            from torch.optim import Adafactor

            return Adafactor(params, **optimizer_kwargs)

        case "adagrad":
            if quantized:
                try:
                    from bitsandbytes.optim import AdaGrad8bit

                    return AdaGrad8bit(params, **optimizer_kwargs)
                except ImportError:
                    warnings.warn(
                        "Package bitsandbytes needed for quantized optimizers, falling back to unquantized optimizer",
                        stacklevel=1,
                    )
            try:
                from bitsandbytes.optim import AdaGrad

                return AdaGrad(params, **optimizer_kwargs)
            except ImportError:
                raise ImportError("Optional dependency bitsandbytes needed for AdaGrad optimizer")

        case "adam":
            if quantized:
                try:
                    from bitsandbytes.optim import Adam8bit

                    return Adam8bit(params, **optimizer_kwargs)
                except ImportError:
                    warnings.warn(
                        "Package bitsandbytes needed for quantized optimizers, falling back to unquantized optimizer",
                        stacklevel=1,
                    )

            from torch.optim.adam import Adam

            return Adam(params, **optimizer_kwargs)
        case "adamw":
            if quantized:
                try:
                    from bitsandbytes.optim import AdamW8bit

                    return AdamW8bit(params, **optimizer_kwargs)
                except ImportError:
                    warnings.warn(
                        "Package bitsandbytes needed for quantized optimizers, falling back to unquantized optimizer",
                        stacklevel=1,
                    )
            from torch.optim.adamw import AdamW

            return AdamW(params, **optimizer_kwargs)

        case "galore":
            if quantized:
                warnings.warn(
                    f"No quantized option available for {optimizer_name}, falling back to unquantized optimizer",
                    stacklevel=1,
                )

            from pytorch_optimizer import GaLore

            return GaLore(params, **optimizer_kwargs)

        case "lamb":
            if quantized:
                try:
                    from bitsandbytes.optim import LAMB8bit

                    return LAMB8bit(params, **optimizer_kwargs)
                except ImportError:
                    warnings.warn(
                        "Package bitsandbytes needed for quantized optimizers, falling back to unquantized optimizer",
                        stacklevel=1,
                    )
            try:
                from bitsandbytes.optim import LAMB

                return LAMB(params, **optimizer_kwargs)
            except ImportError:
                raise ImportError("Optional dependency bitsandbytes needed for LAMB optimizer")

        case "lars":
            if quantized:
                try:
                    from bitsandbytes.optim import LARS8bit

                    return LARS8bit(params, **optimizer_kwargs)
                except ImportError:
                    warnings.warn(
                        "Package bitsandbytes needed for quantized optimizers, falling back to unquantized optimizer",
                        stacklevel=1,
                    )
            try:
                from bitsandbytes.optim import LARS

                return LARS(params, **optimizer_kwargs)
            except ImportError:
                raise ImportError("Optional dependency bitsandbytes needed for LARS optimizer")

        case "lion":
            if quantized:
                try:
                    from bitsandbytes.optim import Lion8bit

                    return Lion8bit(params, **optimizer_kwargs)
                except ImportError:
                    warnings.warn(
                        "Package bitsandbytes needed for quantized optimizers, falling back to unquantized optimizer",
                        stacklevel=1,
                    )
            from pytorch_optimizer import Lion

            return Lion(params, **optimizer_kwargs)

        case "muon":
            if quantized:
                warnings.warn(
                    f"No quantized option available for {optimizer_name}, falling back to unquantized optimizer",
                    stacklevel=1,
                )

            from pytorch_optimizer import Muon

            return Muon(params, **optimizer_kwargs)

        case "sgd":
            if quantized:
                try:
                    from bitsandbytes.optim import SGD8bit

                    return SGD8bit(params, **optimizer_kwargs)
                except ImportError:
                    warnings.warn(
                        "Package bitsandbytes needed for quantized optimizers, falling back to unquantized optimizer",
                        stacklevel=1,
                    )
            from torch.optim.sgd import SGD

            return SGD(params, **optimizer_kwargs)

        case "shampoo":
            if quantized:
                warnings.warn(
                    f"No quantized option available for {optimizer_name}, falling back to unquantized optimizer",
                    stacklevel=1,
                )
            from pytorch_optimizer import Shampoo

            return Shampoo(params, **optimizer_kwargs)

        case "soap":
            if quantized:
                warnings.warn(
                    f"No quantized option available for {optimizer_name}, falling back to unquantized optimizer",
                    stacklevel=1,
                )

            from pytorch_optimizer import SOAP

            return SOAP(params, **optimizer_kwargs)

        case "sophiah":
            if quantized:
                warnings.warn(
                    f"No quantized option available for {optimizer_name}, falling back to unquantized optimizer",
                    stacklevel=1,
                )

            from pytorch_optimizer import SophiaH

            return SophiaH(params, **optimizer_kwargs)

        case "splus":
            if quantized:
                warnings.warn(
                    f"No quantized option available for {optimizer_name}, falling back to unquantized optimizer",
                    stacklevel=1,
                )

            from pytorch_optimizer import SPlus

            return SPlus(params, **optimizer_kwargs)

        case "rmsprop":
            if quantized:
                try:
                    from bitsandbytes.optim import RMSProp8bit

                    return RMSProp8bit(params, **optimizer_kwargs)
                except ImportError:
                    warnings.warn(
                        "Package bitsandbytes needed for quantized optimizers, falling back to unquantized optimizer",
                        stacklevel=1,
                    )
            try:
                from bitsandbytes.optim import RMSProp

                return RMSProp(params, **optimizer_kwargs)
            except ImportError:
                raise ImportError("Optional dependency bitsandbytes needed for RMSProp optimizer")

        case _:
            supported = [
                "adafactor",
                "adagrad",
                "adam",
                "adamw",
                "galore",
                "lamb",
                "lars",
                "lion",
                "muon",
                "sgd",
                "shampoo",
                "soap",
                "sophiah",
                "splus",
                "rmsprop",
            ]
            raise ValueError(f"Unknown optimizer name: {optimizer_name}", f"Supported optimizers: {supported}")
