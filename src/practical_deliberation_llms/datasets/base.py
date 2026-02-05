from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

from ..model import PracticalProblem


class DatasetAdapter(ABC):
    """Base class for dataset adapters returning a normalized DataFrame.

    Adapters should implement `load` and return a `pandas.DataFrame` with
    at least the following columns:

    - `decision_situation`: str
    - `actions`: list[str]
    - `metadata`: dict
    - `problem_uid`: str (optional but recommended)
    """

    # Default configuration for this adapter instance. Subclasses should
    # override this with adapter-specific defaults (e.g. HF config, split),
    # and interpret the merged ``adapter_kwargs`` in ``load``.
    DEFAULT_ADAPTER_KWARGS: Dict[str, Any] = {}

    def __init__(self, dataset_name: str, **adapter_kwargs: Any):
        self.dataset_name = dataset_name

        # Merge class-level defaults with user-provided overrides from YAML
        # or other configuration sources.
        merged: Dict[str, Any] = dict(self.DEFAULT_ADAPTER_KWARGS)
        merged.update(adapter_kwargs or {})
        self.adapter_kwargs = merged

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return the normalized dataset as a DataFrame.

        Expected columns in the returned DataFrame:

        - `decision_situation`: str
        - `actions`: list[str]
        - `metadata`: dict
        - `problem_uid`: str (optional)

        Adapters may omit `problem_uid`; downstream helpers such as
        :func:`make_problem_from_row` will synthesize a deterministic
        identifier when it is missing. Implementations are encouraged to
        provide a stable `problem_uid` where convenient, but are not
        required to do so.
        """


def make_problem_from_row(row: pd.Series) -> PracticalProblem:
    """Convert a normalized DataFrame row into a `PracticalProblem`.

    Expected row keys:

    - `decision_situation`: str
    - `actions`: list[str]
    - `metadata`: dict
    - `problem_uid`: str (optional)

    The function attaches metadata fields to the returned `PracticalProblem`
    instance. These fields line up with the extended dataclass fields defined
    in `model.PracticalProblem`, but we use `setattr` to remain robust while
    the dataclass is being evolved.

    If `problem_uid` is absent, this function synthesizes a deterministic
    identifier based on `metadata['source_dataset']` and
    `metadata['source_id']` (or the row index as a fallback).

    Semantics of identifiers used in experiments
    -------------------------------------------

    - `problem_uid` identifies this *specific* problem instance as it appears
      in an experiment run. For base problems this typically encodes the
      dataset and source row (e.g. ``"daily_dilemmas::42"``).
    - `base_problem_uid` (if present in `metadata`) points to the
      *underlying original* dilemma from which a problem instance was
      derived. For base problems this is often absent/``None``; for
      transformed variants it is expected to equal the base problem's
      `problem_uid` so that metrics can group variants with their baseline.
    """

    decision_situation = row["decision_situation"]
    actions = row["actions"]
    metadata = row.get("metadata", {}) or {}

    # Create or reuse stable problem UID
    problem_uid = row.get("problem_uid")
    if problem_uid is None:
        problem_uid = f"{metadata.get('source_dataset', 'unknown')}::{metadata.get('source_id', row.name)}"

    problem = PracticalProblem(
        decision_situation=decision_situation,
        actions=list(actions),
    )

    # Attach metadata dynamically, so older code that does not yet declare
    # these fields continues to work.
    setattr(problem, "problem_uid", problem_uid)
    setattr(problem, "source_dataset", metadata.get("source_dataset"))
    setattr(problem, "source_id", metadata.get("source_id"))
    setattr(problem, "metadata", metadata)
    setattr(problem, "base_problem_uid", metadata.get("base_problem_uid"))
    setattr(problem, "transformation_type", metadata.get("transformation_type"))
    setattr(
        problem,
        "transformation_params",
        metadata.get("transformation_params", {}),
    )

    return problem
