"""
Pustaka Recommender System – Pipeline Orchestrator
===================================================
Run the full pipeline or individual phases from the project root:

    python main.py               # run ALL phases
    python main.py --phase 6d   # run Phase 6D (KNN) only
    python main.py --phase 6e   # run Phase 6E (Hybrid) only
    python main.py --phase 7    # run Phase 7 (Evaluation) only
    python main.py --list       # show available phases

Usage example (full pipeline):
    python main.py
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# =====================================================
# PHASE REGISTRY
# =====================================================

PHASES = {
    "1":   ("Phase 1  – Merge",                "src/phase1_merge.py"),
    "2":   ("Phase 2  – Cleaning",             "src/phase2_cleaning.py"),
    "3":   ("Phase 3  – EDA",                  "src/phase3_eda.py"),
    "4":   ("Phase 4  – Feature Engineering",  "src/phase4_feature_engineering.py"),
    "5":   ("Phase 5  – Train/Test Split",     "src/phase5_train_test_split.py"),
    "6b":  ("Phase 6B – Content-Based",        "src/phase6b_content_based.py"),
    "6c":  ("Phase 6C – SVD",                  "src/phase6c_svd.py"),
    "6d":  ("Phase 6D – KNN Collaborative",    "src/phase6d_knn.py"),
    "6e":  ("Phase 6E – Hybrid Ensemble",      "src/phase6e_hybrid.py"),
    "7":   ("Phase 7  – Evaluation",           "src/phase7_evaluation.py"),
}

# Default order for running all phases
ALL_ORDER = ["1", "2", "3", "4", "5", "6b", "6c", "6d", "6e", "7"]


# =====================================================
# RUNNER
# =====================================================

def run_phase(phase_key: str) -> bool:
    """
    Run a single phase script.
    Returns True on success, False on failure.
    """
    label, script = PHASES[phase_key]
    script_path   = Path(script)

    if not script_path.exists():
        print(f"\n[SKIP] {label} — script not found: {script}")
        return True                          # non-fatal

    print(f"\n{'=' * 60}")
    print(f"  RUNNING  {label}")
    print(f"{'=' * 60}")

    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,               # stream output to console
        text=True
    )

    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n[OK] {label} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n[FAIL] {label} exited with code {result.returncode}")
        return False


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pustaka Recommender System pipeline runner"
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--phase",
        choices=list(PHASES.keys()),
        help="Run a single phase (e.g. --phase 6d)"
    )

    group.add_argument(
        "--list",
        action="store_true",
        help="List all available phases and exit"
    )

    args = parser.parse_args()

    # ── list ──────────────────────────────────────────
    if args.list:
        print("\nAvailable phases:")
        for key in ALL_ORDER:
            label, script = PHASES[key]
            exists = "OK" if Path(script).exists() else "MISSING"
            print(f"  --phase {key:<4}  {label:<40} [{exists}]")
        return

    # ── single phase ──────────────────────────────────
    if args.phase:
        success = run_phase(args.phase)
        sys.exit(0 if success else 1)

    # ── full pipeline ─────────────────────────────────
    print("\nPUSTAKA RECOMMENDER SYSTEM – FULL PIPELINE")
    print("=" * 60)

    results = {}
    for key in ALL_ORDER:
        ok = run_phase(key)
        results[key] = ok
        if not ok:
            print(f"\n[ABORT] Pipeline stopped at phase {key}.")
            break

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    for key in ALL_ORDER:
        if key not in results:
            status = "SKIPPED"
        else:
            status = "PASS" if results[key] else "FAIL"
        label = PHASES[key][0]
        print(f"  {status:<8}  {label}")

    all_ok = all(results.values())
    print(f"\nPipeline {'COMPLETED SUCCESSFULLY' if all_ok else 'FINISHED WITH ERRORS'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
