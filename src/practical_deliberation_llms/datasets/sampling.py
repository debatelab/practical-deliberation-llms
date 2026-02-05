from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..model import PracticalProblem
from .base import make_problem_from_row
from .registry import load_normalized_dataset


def sample_problems(
    dataset_name: str,
    n_problems: int,
    seed: int,
    adapter_kwargs: Dict[str, Any] | None = None,
) -> List[PracticalProblem]:
    """Sample a fixed-size subset of problems from the given dataset.

    Uses a reproducible random seed and converts rows into `PracticalProblem`
    instances via `make_problem_from_row`.
    """

    df = load_normalized_dataset(dataset_name, adapter_kwargs=adapter_kwargs or {})
    n_total = len(df)
    n = min(n_problems, n_total)

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(n_total, size=n, replace=False)

    sampled_df = df.iloc[sampled_indices].reset_index(drop=True)

    problems: List[PracticalProblem] = []
    for _, row in sampled_df.iterrows():
        problem = make_problem_from_row(row)
        problems.append(problem)

    return problems
