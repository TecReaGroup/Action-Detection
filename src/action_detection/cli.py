"""Application command-line entry points."""

import argparse
import logging
import tomllib

from .logging import configure_logging
from .setting import CONFIG_PATH

DEFAULT_ACTION = "摇摆大拇指"


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be positive")
    return number


def main() -> int:
    """Dispatch training or the live camera preview."""
    parser = argparse.ArgumentParser(description="Single-person hand action recognition")
    command = parser.add_subparsers(dest="command", required=True)
    train = command.add_parser("train", help="Train from action videos")
    train.add_argument("--action", default=DEFAULT_ACTION)
    train.add_argument("--epochs", type=positive_integer, help="Override train.epochs in config")
    train.add_argument("--batch-size", type=positive_integer, default=16)
    preview = command.add_parser("run", help="Open the camera preview")
    preview.add_argument("--camera", type=int, default=0)
    arguments = parser.parse_args()
    configure_logging()
    try:
        if arguments.command == "train":
            from .train import train_model

            epochs = arguments.epochs
            if epochs is None:
                with CONFIG_PATH.open("rb") as stream:
                    section = tomllib.load(stream).get("train")
                if not isinstance(section, dict):
                    raise ValueError(f"Missing [train] section in {CONFIG_PATH}")
                epochs = section.get("epochs")
                if type(epochs) is not int or epochs < 1:
                    raise ValueError("train.epochs must be a positive integer")
            logging.getLogger(__name__).info("Training epochs=%d", epochs)
            train_model(arguments.action, epochs, arguments.batch_size)
            return 0
        if arguments.camera < 0:
            parser.error("Camera index must be nonnegative")
        from .preview import run_preview

        return run_preview(arguments.camera)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupted")
        return 130
    except Exception:
        logging.getLogger(__name__).exception("Command failed")
        return 1
