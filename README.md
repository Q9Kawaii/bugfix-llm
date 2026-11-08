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

## Training

`src/train_qlora.py` uses plain transformers + PEFT and runs on a free Colab/Kaggle T4:

- Base model frozen in 4-bit NF4 (double quantization), fp16 compute.
- LoRA r=16, α=32, dropout 0.05 on all attention and MLP projections.
- Loss on the assistant answer only (the prompt tokens are masked).
- 1 epoch, LR 1e-4 with cosine decay, effective batch 16, paged 8-bit AdamW, gradient checkpointing.
- Checkpoints every 25 steps; re-running the same command resumes after a disconnect.

```bash
python src/train_qlora.py --output-dir runs/qlora-7b
```

## Evaluation

Greedy **pass@1**, test-verified. The model gets a buggy function plus its unit tests, returns a fixed function, and the tests are executed.

- **HumanEvalFix** ([humanevalpack](https://huggingface.co/datasets/bigcode/humanevalpack), Python, 164 hand-written bugs): the main benchmark, out of distribution relative to the training data.
- **Held-out:** 300 synthetic bugs from `data/heldout.jsonl`, from problems never seen in training (in distribution).

```bash
pip install -r requirements.txt
python src/evaluate.py --oracle canonical                       # harness check: 100%
python src/evaluate.py --oracle buggy                           # harness check: 0%
python src/evaluate.py --run-name base-7b                       # base model (GPU)
python src/evaluate.py --benchmark heldout --run-name base-7b-heldout
python src/evaluate.py --run-name ft-7b --adapter runs/qlora-7b/adapter
```

Results are written to `results/<run-name>/summary.json` (including per-bug-type scores) and `samples.jsonl`. `--rescore results/<run>/samples.jsonl` re-scores saved outputs without a GPU.

## Results

| Model | HumanEvalFix pass@1 | Held-out pass@1 |
|---|---|---|
| Qwen2.5-Coder-7B-Instruct (base, 4-bit) | **61.6%** (101/164) | _pending_ |
| + QLoRA bug-fix fine-tune | _pending_ | _pending_ |
