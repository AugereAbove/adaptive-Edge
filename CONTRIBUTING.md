# Contributing

> Checkout scope: The source test suite and CI workflow are excluded from this staged publication. Test commands and test-file references below describe the complete source release; they cannot be run here without restoring that suite.

Install `python -m pip install -e ".[dev,ollama,plotting]"` in a virtual environment, then `pre-commit install` inside your Git checkout.

Before proposing a change, run:

```text
ruff check .
ruff format --check .
mypy src
pytest --cov-fail-under=94.28
python scripts/audit_source.py
python scripts/audit_repository.py
python -m build
```

Source code must contain no comments or docstrings. Explain decisions in Markdown documentation. Use typed names, focused functions and tests to explain behavior. Do not add model downloads, external services or GPU requirements to default tests. HTTP contracts use in-memory transports. Tests use temporary SQLite databases.

Preserve raw measurements. Add migrations rather than rewriting history. New metrics must describe their scale and limitations; never label lexical matching as human-rated quality. Add adapters through the registry and test cancellation, missing usage and unavailable measurements. New policies must define deterministic tie-breaking and enforce hard constraints.

Pull requests should explain the problem, resulting behavior, relevant methodology choices and executed validation. Changes to prediction grouping or cost measurement require accompanying methodology documentation and regression tests.

Use strict mypy for public interfaces. Real-runtime tests require opt-in environment flags and ollama/real_backend/slow markers; default tests stay offline. Freeze calibration, reserve validation for tuning and declare weights/exclusions before evaluation. Never commit fabricated values, model weights, databases or large outputs. Small curated results require provenance, digests and regenerable tables. Run `python -m pip_audit`; scanning is not a security guarantee.
