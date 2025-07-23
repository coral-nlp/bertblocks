"""PolyBert pretraining utilities and implementations.

This module provides comprehensive utilities for pretraining PolyBert models
using masked language modeling (MLM) and other self-supervised objectives.

The pretraining framework includes:
- PyTorch Lightning-based training modules
- Efficient data loading with streaming support
- Dynamic masking for MLM training
- Advanced optimization strategies
- Model compilation support for improved performance

The pretraining setup is designed to be scalable and easily configurable
for different datasets, model sizes, and training regimes.
"""
