from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from torch.optim import Optimizer


def get_optimizer(
    optimizer_name: Literal["sgd", "adam", "adamw", "adafactor", "shampoo", "lion", "sophia", "muon", "galore", "soap"],
    params: list[dict[str, Any]],
    optimizer_kwargs: dict[str, Any],
) -> Optimizer:
    """Instantiate a specific optimizer with params and hyperparameters and return it.

    Args:
        optimizer_name: Name of the optimizer class to instantiate.
        params: Model parameters to be optimized.
        optimizer_kwargs: Optional hyperparameters to pass to optimizer.


    Returns:
        The instantiated optimizer.

    Raises:
        ValueError: If the specified optimizer name is not recognized.

    Supported normalization types:
        - `sgd`
        - `adam`
        - `adamw`
        - `adafactor`
        - `shampoo`
        - `lion`
        - `sophia`
        - `muon`
        - `galore`
        - `soap`

    """
    match optimizer_name:
        case "sgd":
            from torch.optim.sgd import SGD

            return SGD(params, **optimizer_kwargs)
        case "adam":
            from torch.optim.adam import Adam

            return Adam(params, **optimizer_kwargs)
        case "adamw":
            from torch.optim.adamw import AdamW

            return AdamW(params, **optimizer_kwargs)
        case "adafactor":
            from torch.optim import Adafactor

            return Adafactor(params, **optimizer_kwargs)
        case "shampoo":
            from pytorch_optimizer import Shampoo

            return Shampoo(params, **optimizer_kwargs)
        case "lion":
            from pytorch_optimizer import Lion

            return Lion(params, **optimizer_kwargs)
        case "sophiah":
            from pytorch_optimizer import SophiaH

            return SophiaH(params, **optimizer_kwargs)
        case "muon":
            from pytorch_optimizer import Muon

            return Muon(params, **optimizer_kwargs)
        case "galore":
            from pytorch_optimizer import GaLore

            return GaLore(params, **optimizer_kwargs)
        case "soap":
            from pytorch_optimizer import SOAP

            return SOAP(params, **optimizer_kwargs)
        case "splus":
            # see: https://github.com/kozistr/pytorch_optimizer/issues/396
            raise NotImplementedError
        case _:
            supported = ["sgd", "adam", "adamw", "adafactor", "shampoo", "lion", "sophia", "muon", "galore", "soap"]
            raise ValueError(f"Unknown optimizer name: {optimizer_name}", f"Supported optimizers: {supported}")
