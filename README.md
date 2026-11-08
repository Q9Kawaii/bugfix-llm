# bugfix-llm

Fine-tuning **Qwen2.5-Coder-7B-Instruct** with **QLoRA** to fix bugs in Python code, measured by test-verified pass rate on held-out bugs.

> Work in progress.

## Evaluation

**Benchmark:** [HumanEvalFix](https://huggingface.co/datasets/bigcode/humanevalpack) (Python, 164 tasks). Each task gives the model a buggy function and its unit tests. The model returns a fixed function, and the tests are executed. The metric is greedy **pass@1**.

```bash
pip install -r requirements.txt
python src/eval_humanevalfix.py --oracle canonical   # harness check, should be 100%
python src/eval_humanevalfix.py --oracle buggy       # harness check, should be 0%
python src/eval_humanevalfix.py --run-name base-7b   # base model (needs a GPU)
```

Results are written to `results/<run-name>/summary.json` and `samples.jsonl`.

## Results

| Model | HumanEvalFix-Python pass@1 |
|---|---|
| Qwen2.5-Coder-7B-Instruct (base, 4-bit) | _pending_ |
| + QLoRA bug-fix fine-tune | _pending_ |
