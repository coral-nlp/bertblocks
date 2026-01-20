"""Lightning CLI Wrapper entrypoint."""

from lightning.pytorch.callbacks import ThroughputMonitor
from lightning.pytorch.cli import ArgsType, LightningCLI


def cli(args: ArgsType = None) -> None:
    """Set up the main CLI entrypoint."""
    _ = LightningCLI(
        args=args,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        trainer_defaults={
            "callbacks": [
                ThroughputMonitor(
                    batch_size_fn=lambda batch: batch["input_ids"].size(0),
                    length_fn=lambda batch: batch["input_ids"].numel(),
                    window_size=50,
                )
            ],
        },
    )


if __name__ == "__main__":
    cli()
