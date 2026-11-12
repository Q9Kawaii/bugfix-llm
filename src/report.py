"""Write the README results section from results/*/summary.json.

  python src/report.py   # rewrites the block between RESULTS:START / RESULTS:END
"""

import json
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")


def load(run):
    path = os.path.join(RES, run, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cell(s, bold=False):
    if not s:
        return "_pending_"
    txt = f"{s['pass@1']:.1f}% ({s['n_passed']}/{s['n_tasks']})"
    return f"**{txt}**" if bold else txt


def delta(a, b):
    if not (a and b):
        return "—"
    return f"{b['pass@1'] - a['pass@1']:+.1f} pp"


def main():
    base, base_h = load("base-1.5b"), load("base-1.5b-heldout")
    ft, ft_h = load("ft-1.5b"), load("ft-1.5b-heldout")
    ref = load("base-7b")

    lines = [
        "| Model | HumanEvalFix pass@1 | Held-out pass@1 |",
        "|---|---|---|",
        f"| Qwen2.5-Coder-1.5B-Instruct (base) | {cell(base)} | {cell(base_h)} |",
        f"| + LoRA bug-fix fine-tune (ours) | {cell(ft, True)} | {cell(ft_h, True)} |",
        f"| **Change** | **{delta(base, ft)}** | **{delta(base_h, ft_h)}** |",
    ]
    if ref:
        lines.append(f"| *Reference: Qwen2.5-Coder-7B-Instruct (base, 4-bit)* | *{cell(ref)}* | — |")

    if base and ft:
        lines += ["", "HumanEvalFix by bug type (base → fine-tuned):", "",
                  "| Bug type | Base | Fine-tuned |", "|---|---|---|"]
        for k in sorted(base["by_bug_type"]):
            lines.append(f"| {k} | {base['by_bug_type'][k]} | {ft['by_bug_type'].get(k, '—')} |")

    info_path = os.path.join(RES, "lora-1.5b-train.json")
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        lines += ["", f"Training: {info['examples']:,} examples ({info['tokens']:,} tokens), "
                      f"{info['steps']} steps, peak VRAM {info.get('peak_vram_gb', '?')} GB "
                      f"on an RTX 4060 Laptop GPU. Log: `results/lora-1.5b-train.json`."]

    lines += ["", "Greedy decoding, a single sample per bug, tests executed. HumanEvalFix problems "
                  "were removed from the training data (see decontamination above)."]

    readme = os.path.join(ROOT, "README.md")
    with open(readme, encoding="utf-8") as f:
        text = f.read()
    block = "<!-- RESULTS:START -->\n" + "\n".join(lines) + "\n<!-- RESULTS:END -->"
    text = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: block, text, flags=re.S)
    with open(readme, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(block)


if __name__ == "__main__":
    main()
