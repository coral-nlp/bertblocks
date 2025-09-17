"""Lightning CLI Wrapper entrypoint."""

from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.cli import ArgsType, LightningCLI


def cli(args: ArgsType = None) -> None:
    """Set up the main CLI entrypoint."""
    _ = LightningCLI(
        args=args,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        trainer_defaults={
            "callbacks": [
                # ThroughputMonitor(
                #   batch_size_fn=lambda batch: batch["input_ids"].size(0),  # Number of sequences
                #   window_size=50,
                # ),
                # LearningRateMonitor(logging_interval="step", log_momentum=True, log_weight_decay=True),
                # ModelCheckpoint(
                #    save_top_k=1,
                #    monitor="loss/train",
                #    mode="min",
                #    dirpath="./checkpoints",
                #    every_n_train_steps=10_000,
                #    filename="neobert-de-fineweb-{epoch:02d}-{loss/train:.2f}",
                # ),
            ],
            "detect_anomaly": True,
        },
    )


if __name__ == "__main__":
    cli()
