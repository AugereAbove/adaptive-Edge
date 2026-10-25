import ast
import io
import tokenize
from pathlib import Path


def audit(root: Path) -> list[str]:
    violations = []
    for folder in ("src", "tests", "experiments", "scripts"):
        for path in sorted((root / folder).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type == tokenize.COMMENT:
                    violations.append(f"{path.relative_to(root)}:{token.start[0]}: comment")
            for node in ast.walk(ast.parse(source)):
                if isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ) and ast.get_docstring(node):
                    violations.append(f"{path.relative_to(root)}: docstring")
    return violations


if __name__ == "__main__":
    failures = audit(Path(__file__).resolve().parents[1])
    if failures:
        raise SystemExit("\n".join(failures))
    print("Source audit passed: no Python comments or docstrings")
