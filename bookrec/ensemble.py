DEFAULT_MEMBER_COUNT = 3
DEFAULT_FIRST_SEED = 100
DEFAULT_SEED_STEP = 10


def default_seeds(member_count: int) -> list[int]:
    return [
        DEFAULT_FIRST_SEED + index * DEFAULT_SEED_STEP
        for index in range(member_count)
    ]


def resolve_seeds(member_count: int, seeds: list[int] | None) -> list[int]:
    if member_count < 2:
        raise ValueError("An ensemble requires at least two members")
    selected_seeds = seeds if seeds is not None else default_seeds(member_count)
    if len(selected_seeds) != member_count:
        raise ValueError(
            f"Expected {member_count} seeds, received {len(selected_seeds)}"
        )
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Ensemble training seeds must be unique")
    return selected_seeds
