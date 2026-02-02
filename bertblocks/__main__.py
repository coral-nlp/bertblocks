"""Lightning CLI Wrapper entrypoint."""

from lightning.pytorch.cli import ArgsType, LightningCLI


def cli(args: ArgsType = None) -> None:
    """Set up the main CLI entrypoint."""
    _ = LightningCLI(
        args=args,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    cli()
