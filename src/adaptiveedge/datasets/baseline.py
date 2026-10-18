import json
from typing import Any

from adaptiveedge.core.models import Category, TaskDefinition


def baseline_tasks(split: str) -> tuple[TaskDefinition, ...]:
    if split not in {"calibration", "evaluation"}:
        raise ValueError("expected calibration or evaluation")
    calibration = split == "calibration"
    rows: list[TaskDefinition] = []
    sentiments = (
        (
            ("The service was wonderful.", "positive"),
            ("This product broke immediately.", "negative"),
            ("The parcel arrived today.", "neutral"),
        )
        if calibration
        else (
            ("I loved every minute of the concert.", "positive"),
            ("The food tasted terrible.", "negative"),
            ("The meeting starts at noon.", "neutral"),
        )
    )
    for index, (text, reference) in enumerate(sentiments):
        rows.append(
            TaskDefinition(
                id=f"{split}-classification-{index}",
                category=Category.CLASSIFICATION,
                prompt=(
                    "Classify the sentiment. Reply with exactly one label: "
                    f"positive, negative, or neutral. Text: {text}"
                ),
                reference=reference,
                expected_labels=("positive", "negative", "neutral"),
                metrics=("accuracy",),
                dataset="adaptiveedge-authored",
            )
        )
    people = (
        (("Mira", "Oslo"), ("Leon", "Lima"), ("Sana", "Rome"))
        if calibration
        else (("Tessa", "Bern"), ("Niko", "Kyoto"), ("Arun", "Perth"))
    )
    for index, (name, city) in enumerate(people):
        rows.append(
            TaskDefinition(
                id=f"{split}-extraction-{index}",
                category=Category.EXTRACTION,
                prompt=(
                    f"Extract only the city name from this sentence: {name} moved to {city} "
                    "last year. Reply with the city only."
                ),
                reference=city,
                metrics=("normalized_exact_match", "token_f1"),
                dataset="adaptiveedge-authored",
            )
        )
    problems = (
        (
            ("What is 7 times 6?", "42"),
            ("There are 15 apples. 4 are eaten. How many remain?", "11"),
            ("What is (12 + 8) divided by 4?", "5"),
        )
        if calibration
        else (
            ("What is 9 times 8?", "72"),
            ("There are 23 books. 7 are lent out. How many remain?", "16"),
            ("What is (18 + 12) divided by 5?", "6"),
        )
    )
    for index, (problem, reference) in enumerate(problems):
        rows.append(
            TaskDefinition(
                id=f"{split}-reasoning-{index}",
                category=Category.REASONING,
                prompt=f"{problem} Give only the final numeric answer.",
                reference=reference,
                metrics=("numeric_answer",),
                dataset="adaptiveedge-authored",
            )
        )
    fixtures: Any = (
        (
            ("add", "a, b", "return the sum of a and b", [([2, 3], 5), ([-2, 2], 0), ([0, 7], 7)]),
            (
                "larger",
                "a, b",
                "return the larger of a and b",
                [([2, 3], 3), ([-2, -1], -1), ([7, 7], 7)],
            ),
            (
                "even",
                "n",
                "return whether n is even as a bool",
                [([2], True), ([3], False), ([0], True)],
            ),
        )
        if calibration
        else (
            ("subtract", "a, b", "return a minus b", [([9, 3], 6), ([-2, 2], -4), ([7, 7], 0)]),
            (
                "smaller",
                "a, b",
                "return the smaller of a and b",
                [([2, 3], 2), ([-2, -1], -2), ([7, 7], 7)],
            ),
            (
                "odd",
                "n",
                "return whether n is odd as a bool",
                [([2], False), ([3], True), ([0], False)],
            ),
        )
    )
    for index, (name, arguments, instruction, cases) in enumerate(fixtures):
        rows.append(
            TaskDefinition(
                id=f"{split}-code-{index}",
                category=Category.CODE,
                prompt=(
                    f"Write only a Python function def {name}({arguments}): to {instruction}. "
                    "Use exactly one return expression. No imports, loops, assignments, "
                    "explanation or markdown. Allowed calls: abs, min, max, len."
                ),
                reference=json.dumps(
                    {
                        "function": name,
                        "cases": [{"args": args, "expected": expected} for args, expected in cases],
                    }
                ),
                metrics=("code_test_success",),
                dataset="adaptiveedge-authored",
            )
        )
    return tuple(rows)
