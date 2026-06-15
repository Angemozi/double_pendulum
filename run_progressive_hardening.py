#!/usr/bin/env python3
# =============================================================================
# run_progressive_hardening.py
# -----------------------------------------------------------------------------
# Progressive disturbance-hardening curriculum for the Double Pendulum RL "monk".
#
# It chains several training stages at INCREASING disturbance magnitude, each one
# WARM-STARTED from the previous stage's best checkpoint, so every level stays
# learnable (it begins from the prior level's mastery rather than from scratch).
# After each stage it runs a quick evaluation -- disturbances OFF (stillness) and
# ON (recovery) -- and a SAFETY GUARD halts + rolls back if the policy collapsed.
#
# Collapse is detected two ways (halt if EITHER fires):
#   * calm jitter  : with disturbances OFF, mean |w1|+|w2| jumps above a small
#                    threshold -> the monk forgot how to stand still (the real,
#                    magnitude-independent failure signal). PRIMARY.
#   * avgRet100    : training's last avgReturn100 drops below a floor. SECONDARY
#                    (can be naturally low at high magnitude, so it is advisory).
#
# Usage:
#   python run_progressive_hardening.py                 # full run (60M steps/stage)
#   python run_progressive_hardening.py --steps 8000000 # quicker
#   python run_progressive_hardening.py --stages 4.0,4.5,5.0,5.5,6.0
#   python run_progressive_hardening.py --init models/indestructible_best.ckpt
# =============================================================================
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
TRAIN  = ROOT / "build" / "Release" / "dp_train.exe"
EVAL   = ROOT / "build" / "Release" / "dp_eval.exe"
BASE   = ROOT / "configs" / "recovery.yaml"
MODELS = ROOT / "models"
LOGS   = ROOT / "logs"

_RET = re.compile(r"mean return over \d+ episodes = ([-\d.]+)")
_JIT = re.compile(r"\|w1\|\+\|w2\|=([-\d.]+)")
_OK  = re.compile(r"\[SUCCESS\]")


def sh(cmd, capture):
    """Run a command from the project root. Stream when not capturing."""
    cmd = [str(c) for c in cmd]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, capture_output=capture, text=True)


def make_stage_config(magnitude, run_name):
    """recovery.yaml + a disturbMagnitude override (later keys win in the parser)."""
    text = BASE.read_text(encoding="utf-8", errors="replace")
    text += f"\n# --- progressive hardening override ---\nenv.disturbMagnitude: {magnitude}\n"
    p = LOGS / f"_prog_{run_name}.yaml"   # logs/ is git-ignored -> no clutter
    p.write_text(text, encoding="utf-8")
    return p


def last_avgret100(run_name):
    csvp = LOGS / f"{run_name}_stats.csv"
    if not csvp.exists():
        return None
    rows = list(csv.DictReader(csvp.open(newline="")))
    return float(rows[-1]["avgReturn100"]) if rows else None


def best_or_final(run_name):
    b = MODELS / f"{run_name}_best.ckpt"
    return b if b.exists() else (MODELS / f"{run_name}_final.ckpt")


def evaluate(model, config, episodes):
    r = sh([EVAL, "--model", model, "--config", config, "--episodes", episodes], capture=True)
    out = (r.stdout or "") + (r.stderr or "")
    ret = float(m.group(1)) if (m := _RET.search(out)) else None
    jit = float(m.group(1)) if (m := _JIT.search(out)) else None
    return ret, jit, bool(_OK.search(out))


def main():
    ap = argparse.ArgumentParser(description="Progressive disturbance-hardening curriculum.")
    ap.add_argument("--steps", type=int, default=60_000_000, help="training steps per stage")
    ap.add_argument("--stages", default="4.0,5.0,6.0", help="comma-separated disturbMagnitudes")
    ap.add_argument("--init", default=str(MODELS / "indestructible_best.ckpt"),
                    help="checkpoint to warm-start stage 1 from")
    ap.add_argument("--prefix", default="monk_v3", help="run-name prefix")
    ap.add_argument("--episodes", type=int, default=6, help="eval episodes per stage")
    ap.add_argument("--collapse-return", type=float, default=-1000.0,
                    help="halt if training avgRet100 falls below this")
    ap.add_argument("--collapse-jitter", type=float, default=0.30,
                    help="halt if CALM (disturbances off) mean |w1|+|w2| exceeds this (rad/s)")
    args = ap.parse_args()

    for exe in (TRAIN, EVAL):
        if not exe.exists():
            sys.exit(f"ERROR: missing {exe} -- build the Release config first.")
    start = Path(args.init)
    if not start.exists():
        sys.exit(f"ERROR: starting checkpoint not found: {start}")

    mags = [float(x) for x in args.stages.split(",")]
    LOGS.mkdir(exist_ok=True)
    summary = LOGS / "progressive_hardening.csv"
    with summary.open("w", newline="") as f:
        csv.writer(f).writerow(["stage", "magnitude", "run", "trainAvgRet100",
                                "evalReturn_disturbed", "jitter_calm", "success_calm", "status"])

    last_stable = start
    for i, mag in enumerate(mags, 1):
        run_name = f"{args.prefix}_m{str(mag).replace('.', 'p')}"
        print(f"\n{'='*70}\n STAGE {i}/{len(mags)}: disturbMagnitude {mag}"
              f"   (warm-start from {last_stable.name})\n{'='*70}")
        cfg = make_stage_config(mag, run_name)

        # ---- Train (streamed so progress is visible) ------------------------
        tr = sh([TRAIN, "--config", cfg, "--init", last_stable,
                 "--run", run_name, "--steps", args.steps], capture=False)
        if tr.returncode != 0:
            sys.exit(f"ERROR: training process failed at stage {i} (exit {tr.returncode}).")

        # ---- Telemetry: disturbed (recovery) + calm (stillness) -------------
        model = best_or_final(run_name)
        train_ret = last_avgret100(run_name)
        ret_on, _jit_on, _ = evaluate(model, cfg, args.episodes)              # disturbances ON
        sidecar = MODELS / f"{run_name}_config.yaml"                          # disturbances OFF by default
        ret_off, jit_calm, success_calm = evaluate(model, sidecar, args.episodes)

        print(f"\n  [telemetry] trainAvgRet100={train_ret}  disturbedReturn={ret_on}  "
              f"calmJitter={jit_calm} rad/s  SUCCESS(calm)={success_calm}")

        # ---- Safety guard ---------------------------------------------------
        bad_return = (train_ret is not None and train_ret < args.collapse_return)
        bad_still  = (jit_calm   is not None and jit_calm   > args.collapse_jitter)
        status = "COLLAPSE" if (bad_return or bad_still) else "OK"
        with summary.open("a", newline="") as f:
            csv.writer(f).writerow([i, mag, run_name, train_ret, ret_on, jit_calm, success_calm, status])

        if status == "COLLAPSE":
            why = []
            if bad_still:  why.append(f"calm jitter {jit_calm} > {args.collapse_jitter} (forgot stillness)")
            if bad_return: why.append(f"avgRet100 {train_ret} < {args.collapse_return}")
            prev = mags[i - 2] if i > 1 else mag - 1.0
            mid = round((prev + mag) / 2, 2)
            print("\n" + "!" * 70)
            print(f"  POLICY COLLAPSE at stage {i} (magnitude {mag}): {'; '.join(why)}")
            print(f"  Rolling back. Last STABLE checkpoint: {last_stable}")
            print(f"  Suggestion: insert an intermediate magnitude (e.g. {mid}) between "
                  f"{prev} and {mag}, or train this stage longer, then resume:")
            print(f"    python {Path(__file__).name} --init {last_stable} "
                  f"--stages {mid},{mag}")
            print("!" * 70)
            sys.exit(2)

        print(f"  Stage {i} OK -> next stage warm-starts from {model.name}")
        last_stable = model

    print(f"\n{'='*70}\n DONE. The ultimate indestructible monk: {last_stable}\n"
          f" Telemetry log: {summary}\n{'='*70}")


if __name__ == "__main__":
    main()
