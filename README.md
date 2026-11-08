# bugfix-llm

Fine-tuning **Qwen2.5-Coder-7B-Instruct** with **QLoRA** to fix bugs in Python code, measured by test-verified pass rate on held-out bugs.

> Work in progress.

## Training data

`src/build_dataset.py` builds a **test-verified synthetic bug-fix dataset**:

1. **Source:** correct Python solutions with unit tests from [KodCode-V1](https://huggingface.co/datasets/KodCode/KodCode-V1) (Leetcode, Apps, Taco, Code Contests, Algorithm, Data Structure, Evol, Prefill subsets).
2. **Filters:** reliable solutions (≥67% pass rate in KodCode's own checks), self-contained tests, no I/O or heavy libraries, under 2,000 characters.
3. **Decontamination against HumanEvalFix:** KodCode benchmark-similarity < 0.8, no shared function names (1,946 dropped), and no 13-gram overlap with HumanEval prompts or solutions (106 more dropped).
4. **Bug injection:** one AST mutation per example, covering the same six bug types as HumanEvalFix: operator, value, variable or function misuse, plus missing or excess logic. **A mutant is kept only if the original passes its tests and the mutant fails them.**
5. **Balance:** one example per problem, chosen so that every bug type and source is roughly equally represented.

| Split | Examples | Per bug type | Per source |
|---|---|---|---|
| train | 6,000 | ~1,000 each | ~750 each |
| held-out | 300 | — | — |

Each example uses the same chat prompt as the evaluation (`src/prompting.py`): buggy function + tests → fixed function. Full counts are in [`data/stats.json`](data/stats.json).

```bash
python src/build_dataset.py --n-train 6000 --n-heldout 300   # ~15 min on CPU
```

## Evaluation

**Benchmark:** [HumanEvalFix](https://huggingface.co/datasets/bigcode/humanevalpack) (Python, 164 tasks). Each task gives the model a buggy function and its unit tests. The model returns a fixed function, and the tests are executed. The metric is greedy **pass@1**.

```bash
pip install -r requirements.txt
python src/eval_humanevalfix.py --oracle canonical   # harness check, should be 100%
python src/eval_humanevalfix.py --oracle buggy       # harness check, should be 0%
python src/eval_humanevalfix.py --run-name base-7b   # base model (needs a GPU)
```

Results are written to `results/<run-name>/summary.json` and `samples.jsonl`. To re-score saved completions after a harness change, without a GPU:

```bash
python src/eval_humanevalfix.py --rescore results/base-7b/samples.jsonl
```

The base run took about 9 minutes on a free Colab T4. Per-bug-type scores are in each `summary.json`.

## Results

| Model | HumanEvalFix-Python pass@1 |
|---|---|
| Qwen2.5-Coder-7B-Instruct (base, 4-bit) | **61.6%** (101/164) |
| + QLoRA bug-fix fine-tune | _pending_ |
