"""Benchmark evaluation suite for encoder models."""

from bertblocks.benchmarks.__main__ import run_eval
from bertblocks.benchmarks.base import TaskModule

__all__ = ["run_eval", "TaskModule"]
