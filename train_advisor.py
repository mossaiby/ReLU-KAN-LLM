#!/usr/bin/env python3
# train_advisor.py
# Intelligent diagnostic tool for KAN-LLM training.
#
# Features:
#   1. Automatic detection of config.json, logs, and current hyperparameters.
#   2. Training stage classification (with NaN/Inf divergence guard).
#   3. Regression trend, volatility, and stagnation analysis.
#   4. Generalization gap & overfitting detection (train vs. val comparison).
#   5. Hardware & throughput diagnostics (VRAM headroom, tok/s, ETA to target).
#   6. Concrete, copy-paste CLI recommendations for your next training run.
#
# Usage:
#   python train_advisor.py
#   python train_advisor.py --log checkpoints/kan_model_log.jsonl
#   python train_advisor.py --config config.json --target-steps 30000

import argparse
import json
import math
import statistics
import sys
from datetime import timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers & File Parsers
# ---------------------------------------------------------------------------

def read_log(path):
    entries = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def load_config(config_path):
    p = Path(config_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def lin_slope(xs, ys):
    """Least-squares slope of y vs x."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def safe_ppl(loss):
    if loss is None or math.isnan(loss) or math.isinf(loss):
        return float("inf")
    return math.exp(min(loss, 20.0))


# ---------------------------------------------------------------------------
# Diagnostic Logic
# ---------------------------------------------------------------------------

def classify_stage(loss, ppl, vocab_size=32768):
    """Classifies learning stage based on cross-entropy loss relative to vocabulary size."""
    if math.isnan(loss) or math.isinf(loss):
        return ("DIVERGED / NUMERICAL INSTABILITY",
                "CRITICAL: Loss is NaN or Inf. The model has suffered a gradient explosion "
                "or precision underflow. Resume from the previous checkpoint with a lower learning "
                "rate (e.g. 50% lower) or stronger grad clipping.")

    init_loss = math.log(max(vocab_size, 2))
    if loss >= init_loss:
        return ("Stage 0 - Initialization & Sanity Check",
                f"Loss ({loss:.2f}) is at or above random uniform baseline (~{init_loss:.2f}). "
                "Ensure warmup is active and learning rate is not too high.")

    if loss >= 5.2 or ppl >= 180:
        return ("Stage 1 - Vocabulary & Byte/Token Acquisition",
                "The model is learning frequent subwords, whitespace, and basic character groupings. "
                "Loss drops steeply; higher learning rate is appropriate here.")
    if loss >= 4.0 or ppl >= 55:
        return ("Stage 2 - Local Syntax & Grammatical Structure",
                "The model is mastering grammatical agreement, common collocations, and phrase syntax. "
                "This stage represents the bulk of foundational language capability.")
    if loss >= 3.0 or ppl >= 20:
        return ("Stage 3 - Discourse Coherence & Broad Topic Semantics",
                "Sentences are grammatically sound and form coherent paragraphs. Broad facts appear, "
                "though hallucinations and factual inaccuracies are frequent.")
    if loss >= 2.2 or ppl >= 9:
        return ("Stage 4 - Factual Consolidation & Domain Knowledge",
                "Specific domain terminology and facts are being internalized. Gains become slower; "
                "learning rate decay pays off the most in this stage.")
    if loss >= 1.6 or ppl >= 5:
        return ("Stage 5 - Refined Context & Knowledge Saturation",
                "The architecture is extracting the remaining signal from the corpus. Watch closely for "
                "memorization and overfitting if training on smaller datasets.")
    return ("Stage 6 - Compression Floor / Stylistic Convergence",
            "Loss is near the empirical entropy floor for this model capacity. Diminishing returns from "
            "further steps; prioritize larger datasets or larger model dimension over more steps.")


def analyze_generalization(train_entries, val_entries):
    """Examines the gap between training and validation loss."""
    if not val_entries:
        return None, None, None

    latest_val = val_entries[-1]
    matching_train = [t for t in train_entries if t["step"] <= latest_val["step"]]
    if not matching_train:
        return None, None, None

    cur_train = matching_train[-1].get("train_loss")
    cur_val = latest_val.get("val_loss")
    if cur_train is None or cur_val is None:
        return None, None, None

    gap = cur_val - cur_train

    val_status = "stable"
    if len(val_entries) >= 3:
        v_steps = [v["step"] for v in val_entries[-3:]]
        v_losses = [v["val_loss"] for v in val_entries[-3:]]
        v_slope = lin_slope(v_steps, v_losses) * 1000.0
        if v_slope > 0.05:
            val_status = "rising"
        elif v_slope < -0.05:
            val_status = "falling"

    return gap, cur_val, val_status


# ---------------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------------

def analyze(entries, cfg, cli_lr=None, cli_target_steps=None):
    train = [e for e in entries if e.get("train_loss") is not None]
    val = [e for e in entries if e.get("val_loss") is not None]

    if not train:
        print("[Error] No entries with train_loss found in log.")
        return

    latest_entry = entries[-1]
    latest_step = latest_entry.get("step", 0)
    latest_train_loss = latest_entry.get("train_loss", float("nan"))
    latest_lr = cli_lr or latest_entry.get("lr") or cfg.get("training", {}).get("lr")
    target_steps = cli_target_steps or cfg.get("training", {}).get("steps")
    vocab_size = cfg.get("data", {}).get("tokenizer_vocab_size", 32768)

    best_train = min((e["train_loss"] for e in train if not math.isnan(e["train_loss"])), default=float("nan"))
    best_val = min((e["val_loss"] for e in val if not math.isnan(e["val_loss"])), default=None)
    latest_val_loss = val[-1]["val_loss"] if val else None

    # Reference loss for stage evaluation
    ref_loss = latest_val_loss if latest_val_loss is not None else latest_train_loss
    ref_ppl = safe_ppl(ref_loss)

    print("=" * 72)
    print("               KAN-LLM TRAINING & HYPERPARAMETER REPORT                ")
    print("=" * 72)

    # 1. Summary Metrics
    print(f"Current Step:        {latest_step:,}" + (f" / {target_steps:,} ({latest_step / target_steps:.1%})" if target_steps else ""))
    print(f"Active Learning Rate: {latest_lr:.3e}" if latest_lr else "Active Learning Rate: Unknown")

    train_loss_str = f"{latest_train_loss:.4f} (ppl {safe_ppl(latest_train_loss):.2f})" if not math.isnan(latest_train_loss) else "NaN"
    best_train_str = f"{best_train:.4f} (ppl {safe_ppl(best_train):.2f})" if not math.isnan(best_train) else "NaN"
    print(f"Latest Train Loss:   {train_loss_str} | Best Train: {best_train_str}")

    if val:
        latest_val_str = f"{latest_val_loss:.4f} (ppl {safe_ppl(latest_val_loss):.2f})"
        best_val_str = f"{best_val:.4f} (ppl {safe_ppl(best_val):.2f})"
        print(f"Latest Val Loss:     {latest_val_str} | Best Val:   {best_val_str}")
    else:
        print("Validation:          No validation evals recorded yet (eval_interval=0).")

    # Hardware & Speed Stats
    vram = latest_entry.get("vram_mb")
    tok_s = latest_entry.get("throughput_tok_s")
    tokens_seen = latest_entry.get("tokens_seen", 0)

    perf_str = []
    if vram:
        perf_str.append(f"VRAM: {vram:,.0f} MB")
    if tok_s:
        perf_str.append(f"Speed: {tok_s:,.0f} tok/s")
    if tokens_seen:
        perf_str.append(f"Tokens Seen: {tokens_seen / 1e6:,.1f}M")
    if perf_str:
        print("Telemetry:           " + " | ".join(perf_str))

    # ETA Calculation
    if target_steps and latest_step < target_steps and tok_s and tok_s > 0:
        batch_size = cfg.get("training", {}).get("batch_size", 16)
        max_len = cfg.get("model", {}).get("max_len", 256)
        rem_steps = target_steps - latest_step
        rem_tokens = rem_steps * batch_size * max_len
        eta_sec = rem_tokens / tok_s
        print(f"ETA to Completion:   ~{timedelta(seconds=int(eta_sec))} ({rem_steps:,} steps remaining)")

    # 2. Stage Classification
    stage_title, stage_desc = classify_stage(ref_loss, ref_ppl, vocab_size)
    print("-" * 72)
    print(f"[Stage Assessment] {stage_title}")
    print(f"  Description: {stage_desc}")

    # 3. Generalization & Overfitting Check
    gap, cur_val, val_status = analyze_generalization(train, val)
    if gap is not None:
        print("-" * 72)
        print(f"[Generalization Analysis]")
        print(f"  Generalization Gap (Val - Train): {gap:+.4f}")
        if val_status == "rising" and gap > 0.4:
            print("  WARNING: Validation loss is trending UPWARDS while training loss falls. "
                  "The model is overfitting. Options: increase weight decay, add data, "
                  "or stop training early.")
        elif gap > 0.6:
            print("  NOTICE: Noticeable gap between train and val loss. Check for data leakage, "
                  "small validation size, or corpus memorization.")
        elif gap < -0.1:
            print("  HEALTHY: Validation loss is lower than training loss (expected when using EMA "
                  "weights or dropout during training).")
        else:
            print("  HEALTHY: Validation loss is tracking training loss in close sync.")

    # 4. Regression & Volatility Diagnostics
    series = val if len(val) >= 6 else train
    series_name = "validation loss" if len(val) >= 6 else "training loss"
    recs = []
    suggested_lr = latest_lr

    if len(series) >= 6:
        window_len = max(6, len(series) // 3)
        recent = series[-window_len:]
        steps = [e["step"] for e in recent]
        losses = [e.get("val_loss") if "val_loss" in e and e["val_loss"] is not None else e.get("train_loss") for e in recent]

        mean_loss = statistics.mean(losses)
        step_window = max(steps) - min(steps)
        slope = lin_slope(steps, losses)
        slope_per_1k = slope * 1000.0
        rel_slope = (slope_per_1k / max(mean_loss, 1e-6)) * 100.0

        b = mean_loss - slope * (sum(steps) / len(steps))
        residuals = [y - (slope * x + b) for x, y in zip(steps, losses)]
        noise = statistics.pstdev(residuals) / max(mean_loss, 1e-6)

        print("-" * 72)
        print(f"[Recent Trend] Based on last {len(recent)} {series_name} entries over {step_window:,} steps:")
        print(f"  Slope:          {rel_slope:+.2f}% change per 1,000 steps ({slope_per_1k:+.4f} loss/1k)")
        print(f"  Residual Noise: {noise:.2%} (volatility around trendline)")

        if math.isnan(latest_train_loss) or math.isinf(latest_train_loss):
            recs.append(("CRITICAL DIVERGENCE: Loss exploded to NaN/Inf. Revert to earlier checkpoint.",
                         latest_lr * 0.4 if latest_lr else None))
        elif rel_slope > 1.5:
            recs.append(("LOSS IS RISING: Learning rate is too high or gradient clipping threshold is too loose. "
                         "Cut learning rate by ~50%.",
                         latest_lr * 0.5 if latest_lr else None))
        elif rel_slope > 0.2:
            recs.append(("LOSS IS CREEPING UPWARD: The model is destabilizing or drifting. Reduce LR by 20-30%.",
                         latest_lr * 0.75 if latest_lr else None))
        elif noise > 0.05:
            recs.append(("HIGH LOSS OSCILLATION (>5% noise): Loss is fluctuating erratically around the descent curve. "
                         "Lower LR by ~25% or verify gradient clipping (e.g. --grad-clip 1.0).",
                         latest_lr * 0.75 if latest_lr else None))
        elif -0.3 <= rel_slope <= 0.2:
            if target_steps and latest_step >= 0.85 * target_steps:
                recs.append(("NEAR PLATEAU AT SCHEDULE END: You are past 85% of the total schedule. Cosine decay "
                             "has lowered the LR naturally to refine the model. Training is reaching convergence.",
                             latest_lr))
            else:
                recs.append(("PREMATURE PLATEAU (<0.3% descent): Progress has stalled mid-training. Options:\n"
                             "     a) If LR is already low, you may have reached the capacity limit of this model size.\n"
                             "     b) If LR is high, consider decaying LR faster or checking data diversity.",
                             latest_lr * 0.7 if latest_lr else None))
        elif rel_slope < -4.0:
            recs.append(("RAPID HEALTHY DESCENT (>4% descent per 1k): High learning efficiency. Keep current LR and schedule.",
                         latest_lr))
        else:
            recs.append(("HEALTHY STEADY DESCENT: Steady optimization. Current hyperparameters are performing well.",
                         latest_lr))

        if target_steps:
            pct_done = latest_step / target_steps
            if pct_done >= 0.95:
                recs.append((f"Run is {pct_done:.1%} complete. If the loss is still trending downwards strongly, "
                             f"consider extending total steps via `--steps {latest_step + 5000}`.", None))
            elif pct_done < 0.05:
                recs.append((f"Early warmup phase ({pct_done:.1%}). Keep current parameters until at least 10% progress.", None))

    else:
        print("-" * 72)
        print(f"[Recent Trend] Need at least 6 log entries for regression analysis (found {len(series)}).")
        recs.append(("Training just started. Maintain initial schedule until more log points accumulate.", latest_lr))

    # 5. Output Summary Recommendations
    print("-" * 72)
    print("[Actionable Recommendations]")
    for text, rec_lr in recs:
        print(f"  • {text}")
        if rec_lr and rec_lr != latest_lr:
            suggested_lr = rec_lr
            print(f"    ↳ Suggested Target Base LR: {rec_lr:.3e}")

    # Concrete CLI Command Example
    print("-" * 72)
    print("[Next Step CLI Command]")
    next_steps = max(latest_step + 5000, target_steps or (latest_step + 5000))
    lr_flag = f"--lr {suggested_lr:.3e}" if suggested_lr else ""
    print(f"  To resume training with effective settings:")
    print(f"    python train_kan_llm.py {lr_flag} --steps {next_steps}".strip())
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="KAN-LLM Training Diagnostic & Advisor")
    ap.add_argument("--log", type=str, default=None, help="Path to training log (*_log.jsonl)")
    ap.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    ap.add_argument("--lr", type=float, default=None, help="Override active LR")
    ap.add_argument("--target-steps", type=int, default=None, help="Override target total steps")
    args = ap.parse_args()

    cfg = load_config(args.config)

    log_path = args.log
    if log_path is None:
        ckpt_dir = cfg.get("io", {}).get("checkpoint_dir", "checkpoints")
        name = cfg.get("io", {}).get("checkpoint_name", "kan_model")
        default_candidate = Path(ckpt_dir) / f"{name}_log.jsonl"

        if default_candidate.exists():
            log_path = str(default_candidate)
        else:
            candidates = sorted(Path(ckpt_dir).glob("*_log.jsonl"))
            if candidates:
                log_path = str(candidates[-1])
            else:
                local_candidates = sorted(Path(".").glob("*_log.jsonl"))
                if local_candidates:
                    log_path = str(local_candidates[-1])

    if not log_path or not Path(log_path).exists():
        print(f"[Error] Log file not found. Provide path via --log (e.g. --log checkpoints/kan_model_log.jsonl)", file=sys.stderr)
        sys.exit(1)

    entries = read_log(log_path)
    if not entries:
        print(f"[Error] No valid JSON entries could be parsed from {log_path}", file=sys.stderr)
        sys.exit(1)

    analyze(entries, cfg, cli_lr=args.lr, cli_target_steps=args.target_steps)


if __name__ == "__main__":
    main()
