import argparse
from pathlib import Path

from bookrec.ensemble import DEFAULT_MEMBER_COUNT, resolve_seeds
from scripts.explicit.train import train_explicit_model


DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/explicit/mlp_ensemble")


def train_ensemble(
    member_count: int = DEFAULT_MEMBER_COUNT,
    seeds: list[int] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIRECTORY,
    hyperparameters_path: Path | None = None,
) -> list[Path]:
    """Train explicit MLP copies into seed-specific subdirectories."""
    selected_seeds = resolve_seeds(member_count, seeds)
    return [
        train_explicit_model(
            seed=seed,
            output_dir=output_dir / f"seed_{seed}",
            hyperparameters_path=hyperparameters_path,
        )
        for seed in selected_seeds
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-members",
        "--num-ensembles",
        dest="member_count",
        type=int,
        default=DEFAULT_MEMBER_COUNT,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="One unique seed per member. Defaults to 100, 110, 120, ...",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Parent directory for seed_<n> member directories.",
    )
    parser.add_argument("--hyperparameters-path", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        train_ensemble(
            member_count=args.member_count,
            seeds=args.seeds,
            output_dir=args.output_dir,
            hyperparameters_path=args.hyperparameters_path,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
