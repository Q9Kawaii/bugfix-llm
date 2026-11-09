"""QLoRA fine-tuning of Qwen2.5-Coder on the synthetic bug-fix dataset.

Base model frozen in 4-bit NF4; LoRA adapters (r=16) on all attention and MLP
projections. Loss is computed on the assistant answer only. Built on plain
transformers + PEFT so it runs on a free Colab/Kaggle T4 without extra deps.

  # 3-step smoke test (checks memory + prints seconds/step)
  python src/train_qlora.py --output-dir runs/smoke --max-steps 3
  # full run; re-run the same command after a disconnect to resume
  python src/train_qlora.py --output-dir /content/drive/MyDrive/bugfix-llm/qlora-7b
"""

import argparse
import json
import os
import random
import time

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
TRAIN_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "train.jsonl")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(example: dict, tok, max_len: int) -> dict | None:
    """Prompt tokens get label -100 so only the answer is learned.

    Prompt and answer are tokenized separately, exactly as at inference time
    (generation starts right after the prompt tokens).
    """
    msgs = example["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False)
    assert full.startswith(prompt), "chat template: prompt is not a prefix of full text"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    ids = p_ids + a_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(p_ids) + a_ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train-file", default=TRAIN_FILE)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="override epochs (smoke tests)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU, then rerun.")
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)
    from transformers.trainer_utils import get_last_checkpoint

    # ---------------- data
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    with open(args.train_file, encoding="utf-8") as f:
        raw = [json.loads(l) for l in f]
    random.Random(args.seed).shuffle(raw)
    if args.max_examples:
        raw = raw[: args.max_examples]
    data = [e for e in (encode(r, tok, args.max_len) for r in raw) if e]
    n_tok = sum(len(e["input_ids"]) for e in data)
    print(f"examples: {len(data)} (dropped {len(raw) - len(data)} > {args.max_len} tokens), "
          f"tokens: {n_tok:,}, mean {n_tok // len(data)}", flush=True)

    # ---------------- model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    # ---------------- train
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        fp16=True,
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=data,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )

    last = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last:
        print(f"resuming from {last}", flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last)
    minutes = (time.time() - t0) / 60

    # ---------------- save
    adapter_dir = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    steps = trainer.state.global_step
    info = {
        "base_model": args.model,
        "examples": len(data),
        "tokens": n_tok,
        "steps": steps,
        "minutes_this_session": round(minutes, 1),
        "sec_per_step": round(60 * minutes / max(1, steps - (int(last.split("-")[-1]) if last else 0)), 1),
        "args": vars(args),
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(args.output_dir, "train_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps({k: v for k, v in info.items() if k != "log_history"}, indent=2))
    print(f"adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
