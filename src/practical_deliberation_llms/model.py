"model.py"

import dataclasses
from operator import itemgetter


@dataclasses.dataclass
class PracticalProblem:
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
