"""
run_pipeline.py  --  Full data + training pipeline in one command.

ONE-TIME setup:
    python scripts/run_pipeline.py --setup

EACH-TIME refresh (new data + retrain):
    python scripts/run_pipeline.py

Options:
    --setup        Run SID mapping (only needed once)
    --skip-ohlc    Skip OHLC download (use existing CSV)
    --skip-news    Skip news ingestion (use existing parquet)
    --skip-train   Build dataset only, skip model training
    --label        Label strategy: next_day|same_day|forward_3d (default: next_day)
    --start        OHLC start date (default: 2025-09-01)
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from datetime import datetime
from pathlib import Path

PYTHON = sys.executable
SCRIPTS = Path("scripts")
DATA    = Path("data")


def run(cmd: list[str], label: str) -> bool:
    print()
    print("=" * 60)
    print(f"  STEP: {label}")
    print("=" * 60)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = round(time.time() - start, 1)
    if result.returncode != 0:
        print(f"  FAILED ({elapsed}s) — {label}")
        return False
    print(f"  DONE ({elapsed}s) — {label}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup",      action="store_true")
    p.add_argument("--skip-ohlc",  action="store_true")
    p.add_argument("--skip-news",  action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--label",      default="next_day")
    p.add_argument("--start",      default="2025-09-01")
    a = p.parse_args()

    t0    = time.time()
    today = datetime.now().strftime("%Y-%m-%d")
    steps_ok = []
    steps_fail = []

    print()
    print("=" * 60)
    print("  TDWL PREDICTION PIPELINE")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    mode = "SETUP (one-time)" if a.setup else "REFRESH (each-time)"
    print(f"  Mode    : {mode}")
    print("=" * 60)

    def step(cmd, label):
        ok = run(cmd, label)
        (steps_ok if ok else steps_fail).append(label)
        return ok

    # ── STEP 0: SID mapping (one-time only) ──────────────
    if a.setup:
        sid_map = DATA / "processed" / "sid_map.json"
        if sid_map.exists():
            print(f"  sid_map.json already exists — skipping (delete to redo)")
        else:
            if not step([PYTHON, str(SCRIPTS/"find_all_sids.py")],
                        "Build SID map (one-time)"):
                print("SID map failed — cannot continue.")
                return 1

    # ── STEP 1: Fetch OHLC from Mubasher ZIPs ────────────
    if not a.skip_ohlc:
        step([PYTHON, str(SCRIPTS/"fetch_mubasher_ohlc.py"),
              "--start", a.start, "--end", today],
             f"Fetch OHLC ZIPs ({a.start} -> {today})")
    else:
        print("  [SKIPPED] OHLC download")

    # ── STEP 2: Ingest news + announcements ──────────────
    if not a.skip_news:
        step([PYTHON, str(SCRIPTS/"fetch_tdwl_news_announcements.py")],
             "Ingest news + announcements")
    else:
        print("  [SKIPPED] News ingestion")

    # ── STEP 3: Build labelled training dataset ───────────
    ohlc_csv    = str(DATA/"processed"/"ohlc_tdwl.csv")
    news_parq   = str(DATA/"ingested"/"tdwl_news_announcements.parquet")
    real_csv    = str(DATA/"processed"/"real_training_data.csv")
    if not step([PYTHON, str(SCRIPTS/"build_training_from_ohlc.py"),
                 "--ohlc",  ohlc_csv,
                 "--news",  news_parq,
                 "--out",   real_csv,
                 "--label", a.label],
                "Build labelled training dataset"):
        print("Dataset build failed — cannot train.")
        return 1

    # ── STEP 4: Train all models ─────────────────────────
    if not a.skip_train:
        if not step([PYTHON, str(SCRIPTS/"train_all_models.py"),
                     "--data", real_csv],
                    "Train all models (RF + XGB + LSTM)"):
            print("Training failed — skipping evaluation.")
        else:
            # ── STEP 5: Evaluate ─────────────────────────
            step([PYTHON, str(SCRIPTS/"evaluate_all_models.py")],
                 "Evaluate models")
    else:
        print("  [SKIPPED] Training")

    # ── SUMMARY ──────────────────────────────────────────
    total = round(time.time() - t0, 1)
    print()
    print("=" * 60)
    print(f"  PIPELINE COMPLETE in {total}s")
    if steps_ok:
        print(f"  OK   : {len(steps_ok)} steps")
        for s in steps_ok:
            print(f"    + {s}")
    if steps_fail:
        print(f"  FAIL : {len(steps_fail)} steps")
        for s in steps_fail:
            print(f"    x {s}")
    print("=" * 60)
    return 0 if not steps_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
