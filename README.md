# bugfix-llm

Fine-tuning **Qwen2.5-Coder-1.5B-Instruct** with **LoRA** to fix bugs in Python code, measured by test-verified pass rate on bugs the model never saw. Everything (data building, training, evaluation) runs on a single 8 GB laptop GPU.

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

`src/train_qlora.py` uses plain transformers + PEFT:

- Base model frozen in bf16 (pass `--load-4bit` for QLoRA, which is needed for 7B on a 16 GB GPU).
- LoRA r=16, α=32, dropout 0.05 on all attention and MLP projections (18.5M trainable parameters, 1.2% of the model).
- Loss on the assistant answer only (the prompt tokens are masked).
- 1 epoch, LR 1e-4 with cosine decay, effective batch 16, gradient checkpointing.
- Checkpoints every 25 steps; re-running the same command resumes.

Hardware: one **RTX 4060 Laptop GPU (8 GB)**, at about 13.5 s per optimizer step and 5.1 GB peak VRAM.

```bash
python src/train_qlora.py --output-dir runs/lora-1.5b --max-examples 3000 --batch-size 1 --grad-accum 16
```

## Evaluation

Greedy **pass@1**, test-verified. The model gets a buggy function plus its unit tests, returns a fixed function, and the tests are executed.

- **HumanEvalFix** ([humanevalpack](https://huggingface.co/datasets/bigcode/humanevalpack), Python, 164 hand-written bugs): the main benchmark, out of distribution relative to the training data.
- **Held-out:** 300 synthetic bugs from `data/heldout.jsonl`, from problems never seen in training (in distribution).

```bash
pip install -r requirements.txt
python src/evaluate.py --oracle canonical                       # harness check: 100%
python src/evaluate.py --oracle buggy                           # harness check: 0%
python src/evaluate.py --run-name base-1.5b                     # base model (GPU)
python src/evaluate.py --benchmark heldout --run-name base-1.5b-heldout
python src/evaluate.py --run-name ft-1.5b --adapter runs/lora-1.5b/adapter
```

Results are written to `results/<run-name>/summary.json` (including per-bug-type scores) and `samples.jsonl`. `--rescore results/<run>/samples.jsonl` re-scores saved outputs without a GPU.

## Results

<!-- RESULTS:START -->
| Model | HumanEvalFix pass@1 | Held-out pass@1 |
|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct (base) | 20.7% (34/164) | 41.3% (124/300) |
| + LoRA bug-fix fine-tune (ours) | _pending_ | _pending_ |
| **Change** | **—** | **—** |
| *Reference: Qwen2.5-Coder-7B-Instruct (base, 4-bit)* | *61.6% (101/164)* | — |

Greedy decoding, a single sample per bug, tests executed. HumanEvalFix problems were removed from the training data (see decontamination above).
<!-- RESULTS:END -->
