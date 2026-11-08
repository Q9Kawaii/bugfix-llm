"""Build a test-verified Python bug-fix dataset from KodCode.

Pipeline
  1. Load correct solutions + unit tests from KodCode-V1 (selected shards).
  2. Filter: reliable solutions, self-contained tests, moderate length.
  3. Decontaminate against HumanEvalFix (KodCode benchmark similarity,
     entry-point names, 13-gram overlap with HumanEval prompts/solutions).
  4. Strip docstrings, check the original passes its tests.
  5. Inject one bug by AST mutation and keep it only if the tests now FAIL.
  6. Write chat-format train/held-out splits plus stats.

  python src/build_dataset.py --n-train 6000 --n-heldout 300
"""

import argparse
import ast
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from huggingface_hub import hf_hub_download

from execute import program, run_program
from mutate import BUG_TYPES, mutate, strip_docstrings
from prompting import chat

KODCODE = "hf://datasets/KodCode/KodCode-V1/data/train-{:05d}-of-00015.parquet"
SHARDS = [0, 3, 4, 8, 11]  # Leetcode, Apps, Taco, Algorithm, Data_Structure, ...
EXCLUDE_SUBSETS = {"Docs", "Package", "Filter"}
COLS = ["question_id", "subset", "question", "solution", "test", "test_info",
        "gpt_pass_percentage", "benchmark_similarity"]
BAD_TEST = r"fixture|parametrize|class Test|solution\.|monkeypatch|capsys|tmp_path|mock|open\(|unittest"
BAD_SOL = r"input\(|print\(|open\(|sys\.stdin|import (?:numpy|pandas|requests|torch|scipy)|from (?:numpy|pandas)"

def clean_tests(test: str) -> str:
    test = re.sub(r"from\s+solution\s+import\s*\([^)]*\)", "", test)
    lines = [l for l in test.splitlines()
             if not re.match(r"\s*(from\s+solution\s+import|import\s+solution)\b", l)]
    return "\n".join(lines).strip()


# ------------------------------------------------------------- decontamination
def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def ngrams(tokens: list[str], n: int = 13) -> set:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def humaneval_reference():
    path = hf_hub_download("bigcode/humanevalpack", "python/test-00000-of-00001.parquet",
                           repo_type="dataset")
    he = pd.read_parquet(path)
    grams = set()
    for _, r in he.iterrows():
        grams |= ngrams(words(r["prompt"] + r["canonical_solution"]))
    return set(he["entry_point"]), grams


# ------------------------------------------------------------- per-row work
def make_examples(row: dict, seed: int):
    """One caught mutant per bug type that applies to this problem."""
    rng = random.Random(f"{seed}-{row['question_id']}")
    try:
        tree = strip_docstrings(ast.parse(row["solution"]))
        fixed = ast.unparse(tree)
    except Exception:
        return [], "unparseable"
    ep = row["entry_point"]
    if f"def {ep}(" not in fixed:
        return [], "entry point missing"
    tests = clean_tests(row["test"])
    if not run_program(program(fixed, tests), timeout=5)["passed"]:
        return [], "original fails tests"

    out = []
    for bug_type in BUG_TYPES:
        for _ in range(2):  # two tries per type
            buggy = mutate(tree, bug_type, rng)
            if not buggy or buggy == fixed:
                continue
            res = run_program(program(buggy, tests), timeout=5)
            if res["passed"] or "timeout" in res["error"]:
                continue  # tests don't catch it, or it hangs
            out.append({
                "id": row["question_id"],
                "source_subset": row["subset"],
                "bug_type": bug_type,
                "entry_point": ep,
                "buggy_code": buggy,
                "fixed_code": fixed,
                "tests": tests,
                "messages": chat(ep, buggy, tests, fixed),
            })
            break
    return out, "ok" if out else "no caught mutation"


def balanced_pick(options_per_problem: list[list[dict]], seed: int) -> list[dict]:
    """One example per problem, choosing the bug type that is rarest so far."""
    rng = random.Random(seed)
    counts = Counter()
    picked = []
    # Problems with fewer options go first so rare types aren't used up early.
    for opts in sorted(options_per_problem, key=len):
        least = min(counts[o["bug_type"]] for o in opts)
        choice = rng.choice([o for o in opts if counts[o["bug_type"]] == least])
        counts[choice["bug_type"]] += 1
        picked.append(choice)
    return picked


# ------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-heldout", type=int, default=300)
    ap.add_argument("--pool-factor", type=float, default=1.2,
                    help="candidates to try per wanted example")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    stats = Counter()

    df = pd.concat([pd.read_parquet(KODCODE.format(i), columns=COLS) for i in SHARDS],
                   ignore_index=True)
    stats["loaded"] = len(df)

    df = df[~df["subset"].isin(EXCLUDE_SUBSETS)]
    df = df[(df["gpt_pass_percentage"] >= 0.67) & (df["benchmark_similarity"] < 0.8)]
    df = df[df["solution"].str.len().between(80, 2000) & df["test"].str.len().between(50, 2000)]
    df = df[~df["test"].str.contains(BAD_TEST) & ~df["solution"].str.contains(BAD_SOL)]
    df = df.assign(entry_point=df["test_info"].map(
        lambda ti: ti[0]["function_name"] if ti is not None and len(ti) else None))
    df = df[df["entry_point"].notna()].drop_duplicates("question_id")
    stats["after_quality_filters"] = len(df)

    he_names, he_grams = humaneval_reference()
    name_hit = df["entry_point"].isin(he_names)
    gram_hit = df.apply(lambda r: bool(ngrams(words(r["question"] + " " + r["solution"])) & he_grams), axis=1)
    stats["decontam_dropped_name"] = int(name_hit.sum())
    stats["decontam_dropped_13gram"] = int((gram_hit & ~name_hit).sum())
    df = df[~name_hit & ~gram_hit]
    stats["after_decontam"] = len(df)

    # Balance sources: equal cap per subset, then shuffle.
    want = int((args.n_train + args.n_heldout) * args.pool_factor)
    cap = max(1, want // df["subset"].nunique())
    df = df.sample(frac=1, random_state=args.seed).groupby("subset").head(cap)
    rows = df.to_dict("records")
    stats["candidates_tried"] = len(rows)
    print(f"trying {len(rows)} candidates with {args.workers} workers", flush=True)

    options, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for opts, why in pool.map(lambda r: make_examples(r, args.seed), rows):
            stats[f"result: {why}"] += 1
            if opts:
                options.append(opts)
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(rows)} processed, {len(options)} usable", flush=True)
    stats["caught_mutants_by_type"] = dict(Counter(o["bug_type"] for opts in options for o in opts))

    examples = balanced_pick(options, args.seed)
    random.Random(args.seed).shuffle(examples)
    heldout = examples[: args.n_heldout]
    train = examples[args.n_heldout: args.n_heldout + args.n_train]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, split in [("train", train), ("heldout", heldout)]:
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")

    summary = {
        "counts": dict(stats),
        "n_train": len(train),
        "n_heldout": len(heldout),
        "train_bug_types": dict(Counter(e["bug_type"] for e in train)),
        "train_sources": dict(Counter(e["source_subset"] for e in train)),
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
