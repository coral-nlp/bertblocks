from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

SMALL_MODELS = {"bert-base-uncased", "answerdotai/ModernBERT-base"}


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the --small-models command line option."""
    parser.addoption(
        "--small-models",
        action="store_true",
        default=False,
        help="Only run tests for the smallest model of each family.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect parametrized tests whose model is not in SMALL_MODELS when --small-models is set."""
    if not config.getoption("--small-models"):
        return
    skip = []
    for item in items:
        # Parametrized items carry their parameter value in callspecs
        if hasattr(item, "callspec"):
            model = item.callspec.params.get("baseline_model")  # type: ignore[attr-defined]
            if model is not None and model not in SMALL_MODELS:
                skip.append(item)
    for item in skip:
        items.remove(item)
