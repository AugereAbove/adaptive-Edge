import ast
import json
import operator
import sys
from collections.abc import Callable
from typing import Any

BINARY: dict[type[ast.operator], Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
COMPARE: dict[type[ast.cmpop], Callable[..., Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
PURE: dict[str, Callable[..., Any]] = {"abs": abs, "min": min, "max": max, "len": len}


def bounded(value: Any) -> Any:
    if type(value) not in (int, float, bool, str, type(None)):
        raise ValueError("unsupported value")
    if isinstance(value, (int, float)) and not abs(value) <= 1e9:
        raise ValueError("numeric bound exceeded")
    if isinstance(value, str) and len(value) > 1024:
        raise ValueError("string bound exceeded")
    return value


def expression(node: ast.AST, values: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return bounded(node.value)
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        left, right = expression(node.left, values), expression(node.right, values)
        if type(left) not in (int, float, bool) or type(right) not in (int, float, bool):
            raise ValueError("arithmetic requires numbers")
        return bounded(BINARY[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
        value = expression(node.operand, values)
        return bounded(
            not value
            if isinstance(node.op, ast.Not)
            else -value
            if isinstance(node.op, ast.USub)
            else +value
        )
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in COMPARE:
        return COMPARE[type(node.ops[0])](
            expression(node.left, values), expression(node.comparators[0], values)
        )
    if isinstance(node, ast.IfExp):
        return expression(node.body if expression(node.test, values) else node.orelse, values)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in PURE
        and not node.keywords
        and len(node.args) <= 4
    ):
        return bounded(PURE[node.func.id](*(expression(arg, values) for arg in node.args)))
    raise ValueError("expression outside the capability-restricted subset")


def evaluate(source: str, fixture: dict[str, Any]) -> float:
    if len(source) > 16384:
        raise ValueError("source too long")
    if source.strip().startswith("```"):
        source = "\n".join(source.strip().splitlines()[1:-1])
    tree = ast.parse(source)
    if len(list(ast.walk(tree))) > 128 or len(tree.body) != 1:
        raise ValueError("AST bound exceeded")
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != fixture["function"]:
        raise ValueError("expected exactly the requested function")
    if (
        function.decorator_list
        or function.args.defaults
        or function.args.kw_defaults
        or function.args.vararg
        or function.args.kwarg
        or function.args.kwonlyargs
        or function.args.posonlyargs
    ):
        raise ValueError("unsupported function signature")
    if (
        len(function.body) != 1
        or not isinstance(function.body[0], ast.Return)
        or function.body[0].value is None
    ):
        raise ValueError("function must contain a single return expression")
    names = [arg.arg for arg in function.args.args]
    cases = fixture["cases"]
    if not 1 <= len(cases) <= 32:
        raise ValueError("invalid fixture count")
    passed = 0
    for case in cases:
        arguments = [bounded(value) for value in case["args"]]
        actual = expression(function.body[0].value, dict(zip(names, arguments, strict=True)))
        passed += actual == case["expected"]
    return passed / len(cases)


def main() -> None:
    try:
        request = json.loads(sys.stdin.read(32768))
        score = evaluate(request["source"], request["fixture"])
    except Exception:
        score = 0.0
    print(json.dumps({"score": score}))


if __name__ == "__main__":
    main()
