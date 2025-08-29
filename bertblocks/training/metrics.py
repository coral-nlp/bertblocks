from typing import Literal

import torchmetrics


def get_metrics_for_task(
    task: Literal["classification", "token_classification", "question_answering"],
) -> dict[str, torchmetrics.Metric]:
    """Get default metrics for a given task.

    Args:
        task: The task type to get metrics for.

    Returns:
        Dictionary mapping metric names to torchmetrics.Metric instances.
    """
    match task:
        case "classification":
            return {
                "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2),
                "f1": torchmetrics.F1Score(task="multiclass", num_classes=2),
                "precision": torchmetrics.Precision(task="multiclass", num_classes=2),
                "recall": torchmetrics.Recall(task="multiclass", num_classes=2),
            }
        case "token_classification":
            return {
                "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2),
                "f1": torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro"),
                "precision": torchmetrics.Precision(task="multiclass", num_classes=2, average="macro"),
                "recall": torchmetrics.Recall(task="multiclass", num_classes=2, average="macro"),
            }
        case "question_answering":
            return {
                "exact_match": torchmetrics.ExactMatch(task="binary"),
                "f1": torchmetrics.F1Score(task="binary"),
            }
        case _:
            raise ValueError(f"Unknown task: {task}")


def get_metrics(*metric_names: str) -> dict[str, torchmetrics.Metric]:
    """Map string metric names to torchmetrics.Metric instances.

    Args:
        *metric_names: Variable number of metric names as strings.

    Returns:
        Dictionary mapping metric names to torchmetrics.Metric instances.

    Raises:
        ValueError: If an unknown metric name is provided.
    """
    metric_map = {
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2),
        "accuracy_binary": torchmetrics.Accuracy(task="binary"),
        "accuracy_multilabel": torchmetrics.Accuracy(task="multilabel", num_labels=2),
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=2),
        "f1_binary": torchmetrics.F1Score(task="binary"),
        "f1_macro": torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro"),
        "f1_micro": torchmetrics.F1Score(task="multiclass", num_classes=2, average="micro"),
        "precision": torchmetrics.Precision(task="multiclass", num_classes=2),
        "precision_binary": torchmetrics.Precision(task="binary"),
        "precision_macro": torchmetrics.Precision(task="multiclass", num_classes=2, average="macro"),
        "precision_micro": torchmetrics.Precision(task="multiclass", num_classes=2, average="micro"),
        "recall": torchmetrics.Recall(task="multiclass", num_classes=2),
        "recall_binary": torchmetrics.Recall(task="binary"),
        "recall_macro": torchmetrics.Recall(task="multiclass", num_classes=2, average="macro"),
        "recall_micro": torchmetrics.Recall(task="multiclass", num_classes=2, average="micro"),
        "mse": torchmetrics.MeanSquaredError(),
        "mae": torchmetrics.MeanAbsoluteError(),
        "r2": torchmetrics.R2Score(),
        "exact_match": torchmetrics.ExactMatch(),
        "bleu": torchmetrics.BLEUScore(),
        "auroc": torchmetrics.AUROC(task="binary"),
        "auc": torchmetrics.AUROC(task="binary"),
    }

    result = {}
    for name in metric_names:
        if name not in metric_map:
            raise ValueError(f"Unknown metric: {name}. Available metrics: {list(metric_map.keys())}")
        result[name] = metric_map[name]

    return result
