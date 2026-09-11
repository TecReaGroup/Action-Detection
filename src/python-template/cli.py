"""Command-line entry point."""

import argparse
from collections.abc import Sequence

from python_template import __version__
from python_template.config import Settings
from python_template.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(prog="python-template")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the application."""
    parser = build_parser()
    parser.parse_args(argv)

    settings = Settings.from_env()
    logger = configure_logging(settings.log_level)
    logger.info("application_started", extra={"environment": settings.environment})
    return 0
