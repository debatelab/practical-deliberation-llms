"model.py"

import dataclasses
from operator import itemgetter


@dataclasses.dataclass
class PracticalProblem:
    """Representation of a single practical decision problem.

    Core fields:

    - ``decision_situation``: natural language description of the dilemma.
    - ``actions``: list of textual options available to the agent.
    - ``labels``: optional list of short labels ("a", "b", ...) used when
      formatting options for prompts. If omitted, labels are generated
      automatically in ``__post_init__``.

    In experiment code (e.g. under ``experiments/re_sampling_stability``),
    additional attributes are attached dynamically using ``setattr`` to avoid
    over-constraining the dataclass while the schema is still evolving. The
    most important of these are:

    - ``problem_uid``: identifier of this *specific* problem instance as it
      appears in an experiment run (e.g. ``"daily_dilemmas::42"`` or
      ``"daily_dilemmas::42::reverse_options"``).
    - ``base_problem_uid``: identifier of the underlying *original* dilemma
      from which this instance was derived. For base problems this may be
      ``None``; for transformed variants it points back to the base problem's
      ``problem_uid``.
    - ``transformation_type`` / ``transformation_params``: describe how a
      transformed variant was obtained from its base problem (e.g.
      ``"reverse_options"``).
    - ``source_dataset`` / ``source_id`` / ``metadata``: provenance
      information attached by dataset adapters.

    Callers should treat these dynamic attributes as part of the informal
    schema used by experiment scripts and metrics, but not as a stable public
    API yet.
    """

    decision_situation: str
    actions: list[str]
    labels: list[str] | None = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = ["abcdefghijk"[i] for i in range(len(self.actions))]

    def reverse_order(self) -> "PracticalProblem":
        return PracticalProblem(
            decision_situation=self.decision_situation,
            actions=self.actions[::-1],
            labels=self.labels,
        )

    @staticmethod
    def options_list(labels, actions, **kwargs) -> str:
        """Formats options as string for prompt"""
        assert len(labels) == len(actions), "labels and options must have same length"
        foptions = [f"({label}) {option}" for label, option in zip(labels, actions)]
        return "\n".join(foptions)


@dataclasses.dataclass
class Choice:
    label_probs: dict[str, float]
    label: str | None = None
    idx: int | None = None

    def __post_init__(self):
        assert self.label is None and self.idx is None, "label and idx must be None"
        self.label = max(self.label_probs.items(), key=itemgetter(1))[0]
        self.idx = list(self.label_probs.keys()).index(self.label)
