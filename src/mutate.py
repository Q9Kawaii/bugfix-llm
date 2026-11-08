"""Inject a single bug into correct Python code via AST mutation.

Bug types mirror the HumanEvalFix taxonomy so training covers the same
kinds of mistakes, while the code itself comes from unrelated problems.
"""

import ast
import builtins
import copy
import random

BUG_TYPES = [
    "operator misuse",
    "value misuse",
    "variable misuse",
    "missing logic",
    "excess logic",
    "function misuse",
]

BINOP_SWAP = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mod, ast.Mod: ast.FloorDiv, ast.Pow: ast.Mult,
    ast.LShift: ast.RShift, ast.RShift: ast.LShift,
    ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd, ast.BitXor: ast.BitOr,
}
CMP_SWAP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is,
}
FUNC_SWAP = {"min": "max", "max": "min", "any": "all", "all": "any"}
METHOD_SWAP = {
    "append": "extend", "lower": "upper", "upper": "lower",
    "startswith": "endswith", "endswith": "startswith",
    "isdigit": "isalpha", "isalpha": "isdigit", "lstrip": "rstrip",
    "rstrip": "lstrip", "keys": "values", "values": "keys",
    "index": "count", "find": "rfind", "rfind": "find",
}
BUILTINS = set(dir(builtins))


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body.pop(0)
                if not body:
                    body.append(ast.Pass())
    return tree


def _function_nodes(tree):
    """All nodes inside function bodies (never module-level imports etc.)."""
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in fn.body:
                yield from ast.walk(stmt)


def _bodies(tree):
    """(list, index) pairs for every statement slot inside functions."""
    seen = set()
    for node in _function_nodes(tree):
        for field in ("body", "orelse", "finalbody"):
            lst = getattr(node, field, None)
            if isinstance(lst, list) and lst and id(lst) not in seen:
                seen.add(id(lst))
                yield lst
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and id(fn.body) not in seen:
            seen.add(id(fn.body))
            yield fn.body


# ------------------------------------------------------------- mutators
def _operator(tree, rng):
    sites = []
    for n in _function_nodes(tree):
        if isinstance(n, (ast.BinOp, ast.AugAssign)) and type(n.op) in BINOP_SWAP:
            sites.append(lambda n=n: setattr(n, "op", BINOP_SWAP[type(n.op)]()))
        elif isinstance(n, ast.Compare):
            for i, op in enumerate(n.ops):
                if type(op) in CMP_SWAP:
                    sites.append(lambda n=n, i=i: n.ops.__setitem__(i, CMP_SWAP[type(n.ops[i])]()))
        elif isinstance(n, ast.BoolOp):
            sites.append(lambda n=n: setattr(n, "op", ast.Or() if isinstance(n.op, ast.And) else ast.And()))
    return sites


def _value(tree, rng):
    sites = []
    for n in _function_nodes(tree):
        if isinstance(n, ast.Constant):
            v = n.value
            if isinstance(v, bool):
                sites.append(lambda n=n: setattr(n, "value", not n.value))
            elif isinstance(v, int):
                new = 1 if v == 0 else v + rng.choice([-1, 1])
                sites.append(lambda n=n, new=new: setattr(n, "value", new))
            elif isinstance(v, float):
                sites.append(lambda n=n: setattr(n, "value", n.value + 1.0))
    return sites


def _variable(tree, rng):
    sites = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                local.add(n.id)
        local -= BUILTINS
        if len(local) < 2:
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in local:
                others = sorted(local - {n.id})
                sites.append(lambda n=n, o=others: setattr(n, "id", rng.choice(o)))
    return sites


def _missing(tree, rng):
    sites = []
    removable = (ast.If, ast.AugAssign, ast.Expr, ast.Continue, ast.Break)
    for lst in _bodies(tree):
        if len(lst) >= 2:
            for i, s in enumerate(lst):
                if isinstance(s, removable):
                    sites.append(lambda lst=lst, i=i: lst.pop(i))
    for n in _function_nodes(tree):
        if isinstance(n, ast.BoolOp) and len(n.values) >= 2:
            sites.append(lambda n=n: n.values.pop(rng.randrange(len(n.values))))
        if isinstance(n, ast.If) and n.orelse:
            sites.append(lambda n=n: setattr(n, "orelse", []))
    return sites


def _excess(tree, rng):
    sites = []
    for lst in _bodies(tree):
        for i, s in enumerate(lst):
            if isinstance(s, ast.AugAssign) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)):
                sites.append(lambda lst=lst, i=i: lst.insert(i + 1, copy.deepcopy(lst[i])))
    for n in _function_nodes(tree):
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name):
            skip = ast.parse(f"if {n.target.id} == {rng.choice([0, 1])}:\n    continue").body[0]
            sites.append(lambda n=n, skip=skip: n.body.insert(0, skip))
    return sites


def _function(tree, rng):
    sites = []
    for n in _function_nodes(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Name) and f.id in FUNC_SWAP:
            sites.append(lambda f=f: setattr(f, "id", FUNC_SWAP[f.id]))
        elif isinstance(f, ast.Name) and f.id == "sorted":
            sites.append(lambda n=n: _flip_reverse(n))
        elif isinstance(f, ast.Attribute) and f.attr in METHOD_SWAP:
            sites.append(lambda f=f: setattr(f, "attr", METHOD_SWAP[f.attr]))
        elif isinstance(f, ast.Attribute) and f.attr == "sort":
            sites.append(lambda n=n: _flip_reverse(n))
    return sites


def _flip_reverse(call: ast.Call):
    for kw in call.keywords:
        if kw.arg == "reverse":
            kw.value = ast.UnaryOp(op=ast.Not(), operand=kw.value)
            return
    call.keywords.append(ast.keyword(arg="reverse", value=ast.Constant(True)))


MUTATORS = {
    "operator misuse": _operator,
    "value misuse": _value,
    "variable misuse": _variable,
    "missing logic": _missing,
    "excess logic": _excess,
    "function misuse": _function,
}


def mutate(tree: ast.AST, bug_type: str, rng: random.Random) -> str | None:
    """Return mutated source for one random site of `bug_type`, or None."""
    t = copy.deepcopy(tree)
    sites = MUTATORS[bug_type](t, rng)
    if not sites:
        return None
    rng.choice(sites)()
    try:
        return ast.unparse(ast.fix_missing_locations(t))
    except Exception:
        return None
