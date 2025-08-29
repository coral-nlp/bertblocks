from typing import Literal

import torchmetrics


def get_metrics_for_task(
    task: Literal["classification", "token_classification", "question_answering"],
    num_labels: int = 2,
) -> dict[str, torchmetrics.Metric]:
    """Get default metrics for a given task.

    Args:
        task: The task type to get metrics for.
        num_labels: The number of labels to use for the task.

    Returns:
        Dictionary mapping metric names to torchmetrics.Metric instances.
    """
    match task:
        case "classification":
            return {
                "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_labels),
                "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_labels),
                "precision": torchmetrics.Precision(task="multiclass", num_classes=num_labels),
                "recall": torchmetrics.Recall(task="multiclass", num_classes=num_labels),
            }
        case "token_classification":
            return {
                "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_labels),
                "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_labels, average="macro"),
                "precision": torchmetrics.Precision(task="multiclass", num_classes=num_labels, average="macro"),
                "recall": torchmetrics.Recall(task="multiclass", num_classes=num_labels, average="macro"),
            }
        case "question_answering":
            return {
                "exact_match": torchmetrics.ExactMatch(task="binary"),
                "f1": torchmetrics.F1Score(task="binary"),
            }
        case _:
            raise ValueError(f"Unknown task: {task}")
