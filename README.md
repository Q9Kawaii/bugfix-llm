# bugfix-llm

**Teaching a small code LLM to fix bugs with test-verified synthetic data and LoRA, on one laptop GPU.**

| | |
|---|---|
| **Result** | **HumanEvalFix pass@1: 20.7% → 51.2% (+30.5 pp, 2.5×)** for Qwen2.5-Coder-1.5B-Instruct |
| **Regressions** | 50 bugs newly fixed, **0** previously-solved bugs broken |
| **Compute** | 32 min of LoRA training, 6.3 GB VRAM, one RTX 4060 Laptop GPU |
| **Data** | 3,000 auto-generated bug/fix pairs, every one verified by executing unit tests |

The training bugs are generated automatically from problems decontaminated against HumanEvalFix, and every reported score comes from **running the tests**, not from string matching or an LLM judge.

---

## How it works

```mermaid
flowchart LR
    A[KodCode-V1<br/>correct solutions + unit tests] --> B[Filter + decontaminate<br/>vs HumanEval]
    B --> C[AST mutation<br/>6 bug types]
    C --> D{original passes?<br/>mutant fails?}
    D -- yes --> E[(train.jsonl<br/>heldout.jsonl)]
    D -- no --> X[discard]
    E --> F[LoRA fine-tune<br/>answer-only loss]
    F --> G[Generate fix]
    G --> H[Run tests in<br/>sandboxed subprocess]
    H --> I[pass@1]
```

### 1. Data: bugs whose ground truth is a test suite

`src/build_dataset.py` + `src/mutate.py`

- **Source:** about 160k solved Python problems with pytest suites from [KodCode-V1](https://huggingface.co/datasets/KodCode/KodCode-V1).
- **Bug injection by AST mutation**, mirroring the HumanEvalFix taxonomy:

  | Bug type | Example mutation |
  |---|---|
  | operator misuse | `<` → `<=`, `+` → `-`, `and` → `or`, `in` → `not in` |
  | value misuse | `0` → `1`, `n` → `n±1`, `True` → `False`, `[::-1]` → `[::-2]` |
  | variable misuse | a variable read replaced by another in-scope variable |
  | missing logic | drop an `if`/`+=`/call statement, a boolean clause, or an `else` |
  | excess logic | duplicate an update, inject `if i == 0: continue` into a loop |
  | function misuse | `min`↔`max`, `any`↔`all`, `.lower()`↔`.upper()`, flip `sorted(reverse=)` |

- **Test-verified:** a mutant is kept only if the original passes its tests **and** the mutant fails them. That gives no silent no-op mutations and no unlabeled noise.
- **Decontamination against HumanEvalFix:** KodCode benchmark-similarity < 0.8, no shared function names (1,946 dropped), and no 13-gram overlap with any HumanEval prompt or solution (106 dropped).
- **Balanced:** a rarest-type-first assignment gives one example per problem with about 1,000 per bug type and about 750 per source subset. A naive first pass skewed toward variable misuse (1,640 vs 554 excess-logic examples), which matters because excess and missing logic were the base model's weakest types.
- 6,000 train + 300 held-out examples, split by problem.

### 2. Training: LoRA with answer-only loss

`src/train_qlora.py`

- LoRA r=16, α=32 on all attention and MLP projections: **18.5M trainable params (1.2%)**.
- **Loss masked to the assistant answer.** The prompt (buggy code + tests) carries no gradient.
- **Train/inference token parity:** prompt and answer are tokenized separately, so the training prompt tokens are byte-identical to what `generate()` sees (verified in a test).
- bf16 base on Ampere+ (fp16 fallback on a T4), plus `--load-4bit` for QLoRA on larger models.
- 1 epoch over 3,000 examples, cosine LR 1e-4, effective batch 16, gradient checkpointing, checkpoint every 25 steps with auto-resume.

### 3. Evaluation: execute, don't eyeball

`src/evaluate.py` + `src/execute.py`

- **HumanEvalFix** (164 hand-written bugs, out of distribution) and **300 held-out synthetic bugs** (in distribution, unseen problems).
- Each candidate fix runs in an isolated `python -I` subprocess with a timeout. Greedy decoding, pass@1.
- **The harness is validated before any model is scored:** the canonical solutions must score 100% and the buggy code must score 0% on both benchmarks (`--oracle canonical|buggy`).
- `--rescore` re-scores saved generations without a GPU, so harness fixes never require regenerating outputs.

---

## Results

<!-- RESULTS:START -->
| Model | HumanEvalFix pass@1 | Held-out pass@1 |
|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct (base) | 20.7% (34/164) | 41.3% (124/300) |
| + LoRA bug-fix fine-tune (ours) | **51.2% (84/164)** | **92.3% (277/300)** |
| **Change** | **+30.5 pp** | **+51.0 pp** |
| *Reference: Qwen2.5-Coder-7B-Instruct (base, 4-bit)* | *61.6% (101/164)* | — |

HumanEvalFix by bug type (base → fine-tuned):

| Bug type | Base | Fine-tuned |
|---|---|---|
| excess logic | 4/31 | 17/31 |
| function misuse | 3/8 | 5/8 |
| missing logic | 4/33 | 8/33 |
| operator misuse | 3/25 | 16/25 |
| value misuse | 12/44 | 26/44 |
| variable misuse | 8/23 | 12/23 |

Training: 3,000 examples (1,609,222 tokens), 188 steps, peak VRAM 6.27 GB on an RTX 4060 Laptop GPU. Log: `results/lora-1.5b-train.json`.

Greedy decoding, a single sample per bug, tests executed. HumanEvalFix problems were removed from the training data (see decontamination above).
<!-- RESULTS:END -->

### What changed

- **Every bug type improved.** The biggest jumps are on *operator misuse* (3 → 16 of 25) and *excess logic* (4 → 17 of 31): exactly the "small, local edit" behaviour the mutation data rewards.
- **Missing logic stays hardest** (4 → 8 of 33). Deleting a statement is easy to synthesize, but *re-inventing* missing logic needs reasoning that single-mutation data teaches less directly.
- **No regressions:** all 34 bugs the base model solved are still solved.
- The fine-tuned model answers with code only (0 answers with stray prose, vs 3 for the base) and more concisely (median 200 vs 240 characters).

### Limitations

- The training bugs are synthetic, with a single mutation each. Real-world bugs are often multi-line and cross-function.
- HumanEvalFix shows the unit tests in the prompt (the "tests" variant). Scores aren't comparable to docstring-only setups.
- These are single-seed, greedy pass@1 numbers. Variance across seeds isn't measured yet.

---

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python src/evaluate.py --oracle canonical                # harness check: 100%
python src/evaluate.py --oracle buggy                    # harness check: 0%

python src/build_dataset.py --n-train 6000 --n-heldout 300                  # ~15 min, CPU
python src/evaluate.py --run-name base-1.5b                                 # before
python src/evaluate.py --benchmark heldout --run-name base-1.5b-heldout
python src/train_qlora.py --output-dir runs/lora-1.5b --max-examples 3000 \
       --batch-size 1 --grad-accum 16                                       # ~32 min, RTX 4060
python src/evaluate.py --run-name ft-1.5b --adapter runs/lora-1.5b/adapter  # after
python src/evaluate.py --benchmark heldout --run-name ft-1.5b-heldout --adapter runs/lora-1.5b/adapter
python src/report.py                                                        # refresh this README's table
```

```
src/
  build_dataset.py   KodCode -> filter -> decontaminate -> mutate -> verify -> balance
  mutate.py          AST mutation operators (6 bug types)
  prompting.py       one chat format shared by training and evaluation
  train_qlora.py     LoRA / QLoRA trainer (transformers + PEFT)
  evaluate.py        generation + execution-based scoring, oracle checks, rescore
  execute.py         sandboxed subprocess runner + pytest shim
  report.py          writes the results table above from results/*/summary.json
data/                train.jsonl, heldout.jsonl, stats.json
results/             summary.json + every generation (samples.jsonl) for each run
```
