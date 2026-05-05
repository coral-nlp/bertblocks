#!/usr/bin/env python3
"""Evaluate a HuggingFace encoder checkpoint on MTEB.

Builds a SentenceTransformer from the HF checkpoint (Transformer +
Pooling [+ Normalize]), then runs MTEB evaluation on it. This is the
stable, documented path for MTEB v2 and avoids wrestling with ModelMeta
loaders.

Examples:
    # One quick task
    python eval_mteb.py --checkpoint bert-base-uncased --tasks STS17

    # Full English v2 benchmark
    python eval_mteb.py --checkpoint ./my_ckpt --benchmark "MTEB(eng, v2)"

    # E5-style model with query/passage prefixes
    python eval_mteb.py --checkpoint ./my_ckpt --tasks NFCorpus \
        --query-prefix "query: " --passage-prefix "passage: "

    # BertBlocks checkpoint from a private HF repo (LM-pretrained, no
    # similarity fine-tuning -> mean pooling, no normalize is a fair baseline)
    python eval_mteb.py --checkpoint lgienapp/gecco-lm-alibi-rms \
        --benchmark "MTEB(eng, v2)" --pooling mean --no-normalize \
        --hf-token "$HF_TOKEN"

    # Long-context eval. Only meaningful for position encodings that
    # extrapolate: ALiBi works at any length out of the box (the bias is
    # computed at runtime); RoPE is bounded by its trained max_sequence_length
    # because the cos/sin cache is a persistent buffer, so going beyond will
    # fail without rebuilding the cache.
    python eval_mteb.py --checkpoint lgienapp/gecco-lm-alibi-rms \
        --benchmark LongEmbed --pooling mean --no-normalize \
        --max-length 8192 --batch-size 4 --hf-token "$HF_TOKEN"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer, models

# Importing bertblocks registers BertBlocksConfig with the HF AutoModel
# system, which is what `models.Transformer(...)` -> `AutoModel.from_pretrained`
# relies on for custom architectures. Keep this import even if it looks unused.
import bertblocks  # noqa: F401

import mteb
from mteb.cache import ResultCache


# ----------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------


def build_sentence_transformer(
    checkpoint: str,
    pooling: str,
    max_seq_length: int,
    normalize: bool,
    device: str | None,
    query_prefix: str,
    passage_prefix: str,
    hf_token: str | None = None,
) -> SentenceTransformer:
    """Wrap a raw HF encoder checkpoint as a SentenceTransformer.

    This is equivalent to writing modules.json + 1_Pooling/config.json
    next to the checkpoint, but done in-memory so we don't touch disk.
    """
    transformer_kwargs: dict = {"max_seq_length": max_seq_length}
    if hf_token:
        # Pass to both the model and the tokenizer downloads.
        transformer_kwargs["model_args"] = {"token": hf_token}
        transformer_kwargs["tokenizer_args"] = {"token": hf_token}
    transformer = models.Transformer(checkpoint, **transformer_kwargs)

    if pooling == "mean":
        pooler = models.Pooling(
            transformer.get_embedding_dimension(),
            pooling_mode="mean",
        )
    elif pooling == "cls":
        pooler = models.Pooling(
            transformer.get_embedding_dimension(),
            pooling_mode="cls",
        )
    else:
        raise ValueError(f"Unknown pooling mode: {pooling!r} (use 'mean' or 'cls')")

    modules = [transformer, pooler]
    if normalize:
        modules.append(models.Normalize())

    st_kwargs: dict = {"modules": modules}
    if device is not None:
        st_kwargs["device"] = device

    # SentenceTransformer supports named prompts; MTEB automatically maps
    # retrieval queries/passages to the 'query' / 'passage' prompts when
    # those names are defined.
    prompts = {}
    if query_prefix:
        prompts["query"] = query_prefix
    if passage_prefix:
        prompts["passage"] = passage_prefix
    if prompts:
        st_kwargs["prompts"] = prompts

    return SentenceTransformer(**st_kwargs)


# ----------------------------------------------------------------------
# Task selection
# ----------------------------------------------------------------------


def build_tasks(args: argparse.Namespace) -> list:
    if args.benchmark:
        bench = mteb.get_benchmark(args.benchmark)
        return list(bench.tasks)
    return list(mteb.get_tasks(tasks=args.tasks))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to (or HF hub ID of) a HuggingFace encoder checkpoint.",
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--benchmark",
        help="Named MTEB benchmark, e.g. 'MTEB(eng, v2)', 'BEIR', 'MTEB(Multilingual)'.",
    )
    g.add_argument("--tasks", nargs="+", help="Explicit MTEB task names.")

    p.add_argument("--output-folder", default="mteb_results")
    p.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    p.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization.")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--device",
        default=None,
        help="'cuda', 'cpu', or 'cuda:N'. Auto-detected if omitted.",
    )
    p.add_argument(
        "--query-prefix",
        default="",
        help="Prefix prepended to queries on retrieval tasks (e.g. 'query: ').",
    )
    p.add_argument(
        "--passage-prefix",
        default="",
        help="Prefix prepended to corpus docs on retrieval tasks (e.g. 'passage: ').",
    )
    p.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="HuggingFace access token for private/gated checkpoints. "
        "Defaults to $HF_TOKEN or $HUGGINGFACE_HUB_TOKEN.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_sentence_transformer(
        checkpoint=args.checkpoint,
        pooling=args.pooling,
        max_seq_length=args.max_length,
        normalize=not args.no_normalize,
        device=args.device,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        hf_token=args.hf_token,
    )
    print(
        f"  pooling={args.pooling}, normalize={not args.no_normalize}, "
        f"max_length={args.max_length}"
    )

    tasks = build_tasks(args)
    task_names = [t.metadata.name for t in tasks]
    print(f"Evaluating on {len(tasks)} task(s): {task_names}")

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cache = ResultCache(str(output_folder))
    results = mteb.evaluate(
        model,
        tasks,
        cache=cache,
        encode_kwargs={"batch_size": args.batch_size},
    )

    # Group results by task type (Retrieval, STS, Classification, ...)
    # — mirrors how the MTEB leaderboard aggregates scores.
    by_type: dict[str, list[tuple[str, float, str]]] = {}
    summary: dict[str, float] = {}
    for tr in results:
        try:
            score = float(tr.get_score())
        except Exception as e:  # noqa: BLE001
            print(f"  {tr.task_name}: <could not read score: {e}>")
            continue
        summary[tr.task_name] = score
        meta = tr.task.metadata
        task_type = getattr(meta, "type", None) or getattr(meta, "task_type", "Other")
        by_type.setdefault(task_type, []).append(
            (tr.task_name, score, getattr(meta, "main_score", ""))
        )

    print("\n=== Results by task type ===")
    type_averages: dict[str, float] = {}
    for task_type in sorted(by_type):
        entries = sorted(by_type[task_type])
        type_avg = sum(s for _, s, _ in entries) / len(entries)
        type_averages[task_type] = type_avg
        print(f"\n[{task_type}]  mean = {type_avg:.4f}  (n={len(entries)})")
        for name, score, metric in entries:
            print(f"  {name:<42s}  {score:.4f}   ({metric})")

    if type_averages:
        # "MTEB average" on the leaderboard is the mean of per-task scores
        # across all tasks (not the mean of per-type means), so report both.
        overall_mean_tasks = sum(summary.values()) / len(summary)
        overall_mean_types = sum(type_averages.values()) / len(type_averages)
        print("\n=== Overall ===")
        print(f"  Mean over {len(summary)} tasks:       {overall_mean_tasks:.4f}")
        print(f"  Mean over {len(type_averages)} task types:  {overall_mean_types:.4f}")

    summary_path = output_folder / "summary.json"
    summary_payload = {
        "per_task": summary,
        "per_type_mean": type_averages,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
