import re
from collections import Counter
from collections.abc import Callable
from math import isfinite

from adaptiveedge.core.models import TaskDefinition
from adaptiveedge.metrics.code import code_test_success

Metric = Callable[[str, str], float]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def exact_match(output: str, reference: str) -> float:
    return float(output == reference)


def normalized_exact_match(output: str, reference: str) -> float:
    return float(normalize(output) == normalize(reference))


def numeric_answer(output: str, reference: str) -> float:
    candidates = re.findall(r"[-+]?\d+(?:\.\d+)?", output.replace(",", ""))
    try:
        return float(bool(candidates) and float(candidates[-1]) == float(reference))
    except ValueError:
        return 0.0


def token_f1(output: str, reference: str) -> float:
    predicted = Counter(normalize(output).split())
    expected = Counter(normalize(reference).split())
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((predicted & expected).values())
    return 2 * overlap / (sum(predicted.values()) + sum(expected.values()))


def macro_f1(predicted: tuple[str, ...], expected: tuple[str, ...]) -> float:
    if len(predicted) != len(expected) or not expected:
        raise ValueError("nonempty equally sized label sequences required")
    labels = sorted(set(expected) | set(predicted))
    scores = []
    for label in labels:
        tp = sum(p == label and e == label for p, e in zip(predicted, expected, strict=True))
        denominator = predicted.count(label) + expected.count(label)
        scores.append(2 * tp / denominator)
    return sum(scores) / len(scores)


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {
            "code_test_success": code_test_success,
            "numeric_answer": numeric_answer,
            "exact_match": exact_match,
            "accuracy": normalized_exact_match,
            "normalized_exact_match": normalized_exact_match,
            "token_f1": token_f1,
        }

    def register(self, name: str, metric: Metric) -> None:
        if name in self._metrics:
            raise ValueError(f"metric already registered: {name}")
        self._metrics[name] = metric

    def evaluate(self, task: TaskDefinition, output: str) -> dict[str, float]:
        if task.reference is None:
            return {}
        scores = {name: self._metrics[name](output, task.reference) for name in task.metrics}
        if any(not isfinite(score) or not 0 <= score <= 1 for score in scores.values()):
            raise ValueError("quality metrics must return finite scores in [0, 1]")
        return scores

    def validate(self, tasks: tuple[TaskDefinition, ...]) -> None:
        unknown = sorted(
            {name for task in tasks for name in task.metrics if name not in self._metrics}
        )
        if unknown:
            raise ValueError(f"unregistered quality metrics: {', '.join(unknown)}")
