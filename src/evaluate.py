"""Test-verified bug-fix evaluation (greedy pass@1).

The model gets a buggy function plus its unit tests and must return the fixed
function. We execute the tests: pass@1 = share of tasks whose tests pass.

Benchmarks
  humanevalfix  HumanEvalFix-Python, 164 hand-written bugs (main, out-of-distribution)
  heldout       data/heldout.jsonl, 300 synthetic bugs never trained on (in-distribution)

Examples
  python src/evaluate.py --run-name base-7b                                  # base
  python src/evaluate.py --run-name ft-7b --adapter /path/to/adapter         # fine-tuned
  python src/evaluate.py --benchmark heldout --run-name base-7b-heldout
  python src/evaluate.py --oracle canonical   # harness check, no GPU: expect 100%
  python src/evaluate.py --oracle buggy       # harness check, no GPU: expect 0%
  python src/evaluate.py --rescore results/base-7b/samples.jsonl            # re-score saved outputs
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

from execute import program as pytest_program
from execute import run_many
from prompting import chat

HUMANEVALFIX = ("bigcode/humanevalpack", "python/test-00000-of-00001.parquet")
HELDOUT = os.path.join(os.path.dirname(__file__), "..", "data", "heldout.jsonl")
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


# ---------------------------------------------------------------- benchmarks
# Every task is normalised to: task_id, bug_type, entry_point, buggy, tests,
# canonical, plus whatever build_program needs.
def load_tasks(benchmark: str, limit: int | None = None) -> list[dict]:
    if benchmark == "humanevalfix":
        import pandas as pd
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(*HUMANEVALFIX, repo_type="dataset")
        tasks = []
        for t in pd.read_parquet(path).to_dict("records"):
            head = t["import"] + t["declaration"]
            tasks.append({
                **t,
                "buggy": head + t["buggy_solution"],
                "canonical": head + t["canonical_solution"],
                "tests": t["test"],
            })
    else:
        with open(HELDOUT, encoding="utf-8") as f:
            tasks = [{
                "task_id": r["id"],
                "bug_type": r["bug_type"],
                "entry_point": r["entry_point"],
                "buggy": r["buggy_code"],
                "canonical": r["fixed_code"],
                "tests": r["tests"],
            } for r in map(json.loads, f)]
    return tasks[:limit] if limit else tasks


def build_program(benchmark: str, t: dict, code: str) -> str:
    if benchmark == "humanevalfix":
        # Imports and helper functions that come before the target function.
        decl = t["import"] + t["declaration"]
        cut = decl.find(f"def {t['entry_point']}")
        prefix = decl[:cut] if cut != -1 else decl
        test = t["test"]
        if f"check({t['entry_point']})" not in test:
            test += f"\ncheck({t['entry_point']})\n"
        return f"{prefix}\n{code}\n\n{t['test_setup']}\n{test}\n"
    # heldout: the buggy file's imports/definitions go first so helpers the
    # model left out still exist; the model's definitions come later and override.
    return pytest_program(f"{definitions_only(t['buggy'])}\n\n{code}", t["tests"])


def definitions_only(src: str) -> str:
    """Imports, functions and classes only (no top-level calls that could crash)."""
    import ast
    keep = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    tree.body = [n for n in tree.body if isinstance(n, keep)]
    return ast.unparse(tree)


# ---------------------------------------------------------------- scoring
def extract_code(text: str, entry_point: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    code = next((b for b in blocks if f"def {entry_point}" in b),
                blocks[0] if blocks else text)
    # Drop top-level test calls the model echoed back (e.g. `check(fib)`):
    # they would run before the harness defines `check`.
    return "\n".join(l for l in code.splitlines() if not l.startswith("check("))


# ---------------------------------------------------------------- generation
def generate(tasks, model_name, adapter, batch_size, max_new_tokens):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU, then rerun.")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.float16,
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    outputs = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        texts = [
            tok.apply_chat_template(chat(t["entry_point"], t["buggy"], t["tests"]),
                                    tokenize=False, add_generation_prompt=True)
            for t in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1]:]
        outputs += tok.batch_decode(new, skip_special_tokens=True)
        print(f"generated {len(outputs)}/{len(tasks)}", flush=True)
    return outputs


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["humanevalfix", "heldout"], default="humanevalfix")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="LoRA adapter path or HF repo")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only first N tasks")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--oracle", choices=["canonical", "buggy"], default=None,
                    help="skip the model; score known code to test the harness")
    ap.add_argument("--rescore", default=None, metavar="SAMPLES_JSONL",
                    help="re-score saved completions without running the model")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    tasks = load_tasks(args.benchmark, args.limit)
    t0 = time.time()

    if args.rescore:
        with open(args.rescore, encoding="utf-8") as f:
            saved = {r["task_id"]: r["completion"] for r in map(json.loads, f)}
        tasks = [t for t in tasks if t["task_id"] in saved]
        completions = [saved[t["task_id"]] for t in tasks]
        run_name = args.run_name or os.path.basename(os.path.dirname(args.rescore))
    elif args.oracle:
        field = "canonical" if args.oracle == "canonical" else "buggy"
        completions = [f"```python\n{t[field]}\n```" for t in tasks]
        run_name = args.run_name or f"oracle-{args.oracle}-{args.benchmark}"
    else:
        completions = generate(tasks, args.model, args.adapter,
                               args.batch_size, args.max_new_tokens)
        run_name = args.run_name or (args.adapter or args.model).rstrip("/").split("/")[-1]

    codes = [extract_code(c, t["entry_point"]) for c, t in zip(completions, tasks)]
    results = run_many([build_program(args.benchmark, t, c) for t, c in zip(tasks, codes)])

    by_type = defaultdict(lambda: [0, 0])
    for t, r in zip(tasks, results):
        by_type[t["bug_type"]][0] += r["passed"]
        by_type[t["bug_type"]][1] += 1

    n_pass = sum(r["passed"] for r in results)
    names = {"humanevalfix": "HumanEvalFix-Python (tests in prompt)",
             "heldout": "Held-out synthetic bugs (data/heldout.jsonl)"}
    summary = {
        "run_name": run_name,
        "benchmark": f"{names[args.benchmark]}, greedy, pass@1",
        "model": args.model,
        "adapter": args.adapter,
        "oracle": args.oracle,
        "n_tasks": len(tasks),
        "n_passed": n_pass,
        "pass@1": round(100 * n_pass / len(tasks), 2),
        "by_bug_type": {k: f"{p}/{n}" for k, (p, n) in sorted(by_type.items())},
        "minutes": round((time.time() - t0) / 60, 1),
    }

    out = os.path.join(args.out_dir, run_name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "samples.jsonl"), "w", encoding="utf-8") as f:
        for t, c, code, r in zip(tasks, completions, codes, results):
            f.write(json.dumps({"task_id": t["task_id"], "bug_type": t["bug_type"],
                                "passed": r["passed"], "error": r["error"],
                                "completion": c, "extracted": code}) + "\n")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
