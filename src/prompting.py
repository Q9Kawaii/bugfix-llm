"""One prompt format shared by training data and evaluation."""

SYSTEM = "You are an expert Python programmer who fixes bugs precisely."


def user_prompt(entry_point: str, buggy_code: str, tests: str) -> str:
    return (
        f"Fix the bug(s) in `{entry_point}`.\n\n"
        f"```python\n{buggy_code.strip()}\n```\n\n"
        f"The fixed function must pass these tests:\n"
        f"```python\n{tests.strip()}\n```\n\n"
        f"Return only the complete fixed function in one ```python code block."
    )


def assistant_answer(fixed_code: str) -> str:
    return f"```python\n{fixed_code.strip()}\n```"


def chat(entry_point: str, buggy_code: str, tests: str, fixed_code: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt(entry_point, buggy_code, tests)},
    ]
    if fixed_code is not None:
        msgs.append({"role": "assistant", "content": assistant_answer(fixed_code)})
    return msgs
