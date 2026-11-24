"""Command line interface."""

import argparse
import math
from pathlib import Path

from .config import read_config


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def duration(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Hours must be finite and positive")
    return result


def training_options(command, default_hours, default_batch):
    command.add_argument("--cache", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--batch", type=positive_int, default=default_batch)
    command.add_argument("--workers", type=int, default=0)
    command.add_argument("--device", default="auto")
    command.add_argument("--seed", type=int, default=1234)
    command.add_argument("--hours", type=duration, default=default_hours)
    command.add_argument("--save-every", type=positive_int, default=500)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="morphoflow-ihc")
    sub = parser.add_subparsers(dest="command", required=True)

    preparation = sub.add_parser("prepare", help="Build a cache from separate training H&E and IHC pools")
    preparation.add_argument("--manifest", type=Path, required=True)
    preparation.add_argument("--config", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)

    flow = sub.add_parser("train", help="Train region, RGB flow, and PatchNCE objectives")
    training_options(flow, 20.0, 4)
    flow.add_argument("--config", type=Path, required=True)
    flow.add_argument("--epochs", type=positive_int, default=25)
    flow.add_argument("--steps", type=positive_int)
    flow.add_argument("--resume", type=Path)

    policy = sub.add_parser("train-rl", help="Optimize the bounded velocity policy at one time step")
    training_options(policy, 6.0, 1)
    policy.add_argument("--checkpoint", type=Path, required=True)
    policy.add_argument("--groups", type=positive_int, default=3000)
    policy.add_argument("--advantage-scale", type=duration, help="Explicit PPO advantage normalization scale")
    policy.add_argument("--policy-lr", type=duration, help="Explicit policy learning rate")

    prediction = sub.add_parser("infer", help="Generate native-size IHC from H&E and a marker")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--image", type=Path, required=True)
    prediction.add_argument("--marker", required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--maps", type=Path)
    prediction.add_argument("--mpp", type=float)
    prediction.add_argument("--device", default="auto")
    prediction.add_argument("--seed", type=int, default=1234)

    assessment = sub.add_parser("evaluate", help="Evaluate a saved prediction against held-out IHC")
    assessment.add_argument("--prediction", type=Path, required=True)
    assessment.add_argument("--reference", type=Path, required=True)
    assessment.add_argument("--marker", required=True)
    assessment.add_argument("--config", type=Path, required=True)
    assessment.add_argument("--output", type=Path)
    assessment.add_argument("--same-section", action="store_true", help="Enable pixel scoring for verified aligned same-section data")

    args = parser.parse_args(argv)
    if getattr(args, "workers", 0) < 0:
        parser.error("--workers must be nonnegative")
    if args.command == "infer" and args.output.suffix.lower() != ".png":
        parser.error("Inference output must have a .png extension")
    if args.command == "infer" and args.maps and args.maps.resolve() == args.output.resolve():
        parser.error("Inference maps and image must have different output paths")

    try:
        if args.command == "prepare":
            from .data import prepare
            count = prepare(args.manifest, args.output, read_config(args.config))
            print(f"Prepared {count} training IHC images and an H&E pool in {args.output}")
        else:
            from . import engine
            if args.command == "train":
                engine.train_flow(args, read_config(args.config))
            elif args.command == "train-rl":
                engine.train_policy(args)
            elif args.command == "infer":
                engine.infer(args)
            else:
                engine.evaluate(args, read_config(args.config))
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
