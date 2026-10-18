from typing import Protocol

from adaptiveedge.core.models import Category, TaskDefinition


class DatasetProvider(Protocol):
    def tasks(self) -> tuple[TaskDefinition, ...]: ...


def basic_suite() -> tuple[TaskDefinition, ...]:
    examples = (
        (
            "sentiment",
            Category.CLASSIFICATION,
            "Classify as positive or negative: I love it.",
            "positive",
        ),
        ("extract", Category.EXTRACTION, "Return only the name: Ada wrote the program.", "Ada"),
        ("arithmetic", Category.REASONING, "Return only the answer: 6 * 7", "42"),
        (
            "summary",
            Category.SUMMARIZATION,
            "Summarize in one sentence: It rained. The match was delayed because of the rain.",
            "Rain delayed the match.",
        ),
        (
            "code",
            Category.CODE,
            "Write only Python code defining add(a, b) returning their sum.",
            "def add(a, b):\n    return a + b",
        ),
    )
    return tuple(
        TaskDefinition(
            id=identifier,
            category=category,
            prompt=prompt,
            reference=reference,
            metrics=("token_f1",)
            if category == Category.SUMMARIZATION
            else ("exact_match",)
            if category == Category.CODE
            else ("normalized_exact_match",),
            dataset="adaptiveedge-functional",
            dataset_version="1",
        )
        for identifier, category, prompt, reference in examples
    )
