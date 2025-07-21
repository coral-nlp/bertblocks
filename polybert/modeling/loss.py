from typing import Literal

from torch import nn


def get_loss_function(
    problem_type: Literal["regression", "single_label_classification", "multi_label_classification"] | None,
) -> nn.Module:
    """Return the applicable loss function for a given problem type.

    Args:
        problem_type: The type of problem ('regression', 'single_label_classification',
                     'multi_label_classification')

    Returns:
        The appropriate loss function module.

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
        raise ValueError(f"Unknown problem type '{problem_type}'. " f"Supported types: {', '.join(supported_types)}")
