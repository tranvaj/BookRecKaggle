import argparse
from pathlib import Path

from bookrec.ensemble import DEFAULT_MEMBER_COUNT, resolve_seeds
from bookrec.implicit.model import MODEL_REGISTRY
from scripts.implicit.train import train_implicit_model


def train_ensemble(
    model_name: str = "mlp",
    member_count: int = DEFAULT_MEMBER_COUNT,
    seeds: list[int] | None = None,
    output_dir: Path | None = None,
    hyperparameters_path: Path | None = None,
) -> list[Path]:
    """Train independent copies of one architecture into seed directories."""
    selected_seeds = resolve_seeds(member_count, seeds)
    ensemble_directory = output_dir or Path(
        f"artifacts/implicit/{model_name}_ensemble"
    )
    checkpoints = []
    for seed in selected_seeds:
        checkpoints.append(
            train_implicit_model(
                model_name=model_name,
                seed=seed,
                output_dir=ensemble_directory / f"seed_{seed}",
                hyperparameters_path=hyperparameters_path,
            )
        )
    return checkpoints


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_REGISTRY, default="mlp")
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
        help="Parent directory for seed_<n> member directories.",
    )
    parser.add_argument("--hyperparameters-path", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        train_ensemble(
            model_name=args.model,
            member_count=args.member_count,
            seeds=args.seeds,
            output_dir=args.output_dir,
            hyperparameters_path=args.hyperparameters_path,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
