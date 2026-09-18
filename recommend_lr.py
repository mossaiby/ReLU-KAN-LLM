#!/usr/bin/env python3
# recommend_lr.py
# Reads the JSONL training log written by train_kan_llm.py and:
#   1. Classifies the current training stage from loss/perplexity heuristics
#      (token/grammar acquisition -> syntax -> semantics/facts -> refinement).
#   2. Recommends a learning-rate adjustment based on the recent loss trend,
#      noise level, and progress through the schedule.
#
# Usage:
#   python recommend_lr.py [--log checkpoints/kan_model_log.jsonl]
#                          [--lr CURRENT_LR] [--target-steps N]
#
# Thresholds are heuristics calibrated for a ~32k vocab (random-init loss
# = ln(32768) ~ 10.4); treat them as a compass, not ground truth.

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def read_log(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def lin_slope(xs, ys):
    # Least-squares slope of y vs x.
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def classify_stage(loss, ppl):
    # Heuristic stage map (val loss / val ppl on a 32k-vocab LM).
    if loss >= 5.5 or ppl >= 250:
        return ("Stage 1 - vocabulary & character/byte structure acquisition",
                "Model is learning token identities. High LR is usually fine here; "
                "expect volatile losses.")
    if loss >= 4.5 or ppl >= 90:
        return ("Stage 2 - grammar & local syntax formation",
                "Local word order and grammar are locking in. Steady descent; "
                "this is where most of the compute should go.")
    if loss >= 3.4 or ppl >= 30:
        return ("Stage 3 - syntax & discourse coherence",
                "Sentences become globally coherent. Factual content starts "
                "appearing but is not yet reliable.")
    if loss >= 2.3 or ppl >= 10:
        return ("Stage 4 - semantic understanding & fact acquisition",
                "Facts and domain knowledge are being absorbed. LR decay now "
                "pays off most; consider lower LR if noisy.")
    if loss >= 1.6 or ppl >= 5:
        return ("Stage 5 - knowledge consolidation",
                "Gains slow; the model refines what it knows. Watch for "
                "memorization/overfitting on small corpora.")
    return ("Stage 6 - refinement / style lock-in",
            "Marginal returns from data; style and surface form dominate "
            "remaining loss. Prefer more data over more steps.")


def analyze(entries, current_lr, target_steps):
    val = [e for e in entries if e.get("val_loss") is not None]
    train = [e for e in entries if e.get("train_loss") is not None]
    if len(entries) < 5:
        print("Not enough log entries yet (need >= 5). Keep training and re-run.")
        return

    last = val[-1] if val else train[-1]
    best_val = min((e["val_loss"] for e in val), default=None)
    best_ppl = math.exp(min(best_val, 20.0)) if best_val is not None else None
    loss_for_stage = best_val if best_val is not None else last.get("train_loss", float("nan"))
    ppl_for_stage = best_ppl if best_ppl is not None else math.exp(min(loss_for_stage, 20.0))

    print("=" * 70)
    print(f"Log entries: {len(entries)} | val evals: {len(val)}")
    print(f"Latest: step {last.get('step')} | train loss {last.get('train_loss', float('nan')):.4f}"
          + (f" | val loss {last.get('val_loss'):.4f} (ppl {last.get('val_ppl'):.2f})" if val else ""))
    print(f"Best val loss: {best_val:.4f} (ppl {best_ppl:.2f})" if best_val is not None else "No val evals yet.")
    print("=" * 70)

    stage, note = classify_stage(loss_for_stage, ppl_for_stage)
    print(f"\n[Stage] {stage}")
    print(f"        {note}")

    # ---- LR recommendation ------------------------------------------------
    recs = []
    series = val if len(val) >= 6 else train
    if len(series) >= 6:
        recent = series[-max(6, len(series) // 3):]
        steps = [e["step"] for e in recent]
        losses = [e.get("val_loss") or e.get("train_loss") for e in recent]
        mean_loss = statistics.mean(losses)
        slope_per_1k = lin_slope(steps, losses) * 1000.0
        rel_slope = slope_per_1k / max(mean_loss, 1e-6) * 100.0  # % change per 1k steps
        window = max(steps) - min(steps)
        # Detrended residual noise: how much the curve wiggles around its
        # fitted line (a steady descent is NOT noise; oscillation is).
        slope = lin_slope(steps, losses)
        b = mean_loss - slope * (sum(steps) / len(steps))
        residuals = [y - (slope * x + b) for x, y in zip(steps, losses)]
        noise = statistics.pstdev(residuals) / max(mean_loss, 1e-6)

        print(f"\n[Trend] last {len(recent)} evals over {window} steps: "
              f"{rel_slope:+.2f}% loss change per 1k steps, residual noise {noise:.2%}")

        if rel_slope > +1.0:
            recs.append(("LOSS IS RISING - LR is almost certainly too high for the "
                         "current stage.", current_lr * 0.5 if current_lr else None))
        elif noise > 0.05:
            recs.append(("Loss oscillates strongly around its trend (residual noise > 5%). Reduce "
                         "LR by ~30-50% or check grad clipping / data quality. Spikes also "
                         "happen in Stage 1-2, so combine with the stage info above.",
                         current_lr * 0.6 if current_lr else None))
        elif rel_slope > -0.5:
            recs.append(("Near-plateau (<0.5% improvement per 1k steps). Options: decay LR "
                         "toward min_lr if late in schedule; otherwise the model may be data-"
                         "saturated - a new/larger data source is worth more than more steps.",
                         current_lr * 0.8 if current_lr else None))
        else:
            recs.append(("Healthy descent. Keep the current LR and schedule.",
                         current_lr))

        # Schedule-position sanity check.
        step_now = series[-1]["step"]
        if target_steps and step_now > 0.8 * target_steps and current_lr:
            recs.append((f"Past 80% of the schedule ({step_now}/{target_steps}): cosine decay "
                         "should have LR near min_lr. If the curve is still dropping fast, "
                         "consider extending --steps instead of stopping.",
                         None))
    else:
        print("\n[Trend] Not enough evals for trend analysis yet.")
        if current_lr:
            recs.append(("Too early to judge - keep warmup + current LR until you have "
                         "several eval points.", current_lr))

    print("\n[LR recommendation]")
    for text, lr in recs:
        print(f"  - {text}")
        if lr:
            print(f"    -> suggested lr: {lr:.3e}")
    print()


def main():
    ap = argparse.ArgumentParser(description="LR recommendation & training-stage analyzer")
    ap.add_argument("--log", type=str, default=None, help="Path to *_log.jsonl")
    ap.add_argument("--lr", type=float, default=None, help="Current base LR (from your config)")
    ap.add_argument("--target-steps", type=int, default=None)
    args = ap.parse_args()

    log_path = args.log
    if log_path is None:
        candidates = sorted(Path("checkpoints").glob("*_log.jsonl"))
        if not candidates:
            print("No log file given and no checkpoints/*_log.jsonl found.", file=sys.stderr)
            sys.exit(1)
        log_path = str(candidates[-1])
    entries = read_log(log_path)
    if not entries:
        print(f"No usable entries in {log_path}", file=sys.stderr)
        sys.exit(1)
    analyze(entries, args.lr, args.target_steps)


if __name__ == "__main__":
    main()
