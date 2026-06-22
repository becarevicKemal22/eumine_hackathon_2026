"""
scoring_program/evaluate.py — Scoring script for EuMINe DataBridge Hackathon 2026.

Expects two arguments:
    sys.argv[1] = input_dir  (contains res/predictions_test.json and ref/test_labels.json)
    sys.argv[2] = output_dir (write scores.json here)

Scoring formula
---------------
For each property p in {formation_energy_per_atom, band_gap}:

    score_p = normalize_score(MAE_p, BASELINE_MAE_p)

where normalize_score maps:
    MAE == baseline  ->  10 pts per property
    MAE == 0         ->  20 pts per property  (theoretical maximum)
    MAE -> infinity  ->   0 pts per property  (floor)

Total performance_score = score_ef + score_bg  (0 – 40 pts)

After running baseline_model.py (Step 1.2), replace the BASELINE_MAE_* constants
with the actual values from baseline_constants.py:

    from baseline_constants import BASELINE_MAE_EF, BASELINE_MAE_BG

"""

import sys
import json

# ---------------------------------------------------------------------------
# Normalization constants — update after running hackathon/baseline_model.py
# ---------------------------------------------------------------------------
#BASELINE_MAE_EF = 0.25   # eV/atom
#BASELINE_MAE_BG = 0.80   # eV
#PERFECT_MAE     = 0.01   # theoretical floor for normalization
#
BASELINE_MAE_EF = 0.2712   # eV/atom — formation energy
BASELINE_MAE_BG = 0.5003   # eV       — band gap
PERFECT_MAE     = 0.01           # theoretical minimum for scoring normalization


def normalize_score(mae: float, baseline_mae: float, perfect_mae: float = PERFECT_MAE) -> float:
    """
    Map MAE to a 0–20 score per property (10 = matches baseline, 20 = perfect).
    Total performance_score = score_ef + score_bg, range 0–40.

    Below baseline (MAE >= baseline_mae): score in [0, 10]
    Above baseline (MAE <  baseline_mae): score in (10, 20]
    """
    if mae >= baseline_mae:
        return max(0.0, 10.0 * (1.0 - (mae - baseline_mae) / baseline_mae))
    else:
        denom = baseline_mae - perfect_mae
        if denom <= 0:
            return 20.0
        return 10.0 + 10.0 * (baseline_mae - mae) / denom


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: evaluate.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir  = sys.argv[1]
    output_dir = sys.argv[2]

    # ── Load predictions ──────────────────────────────────────────────────────
    pred_path = f"{input_dir}/res/predictions_test.json"
    gt_path   = f"{input_dir}/ref/test_labels.json"

    try:
        with open(pred_path) as f:
            pred = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: predictions file not found at {pred_path}")
        sys.exit(1)

    try:
        with open(gt_path) as f:
            gt = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: ground-truth labels not found at {gt_path}")
        sys.exit(1)

    # ── Build lookup maps ─────────────────────────────────────────────────────
    gt_map   = {e["material_id"]: e for e in gt["labels"]}
    pred_map = {e["material_id"]: e for e in pred["predictions"]}

    missing = set(gt_map.keys()) - set(pred_map.keys())
    if missing:
        print(f"WARNING: {len(missing)} materials missing from predictions; scoring as MAE=9999")

    # ── Compute per-property MAE ───────────────────────────────────────────────
    ef_errors: list = []
    bg_errors: list = []

    for mid, gt_entry in gt_map.items():
        if mid in pred_map:
            ef_errors.append(abs(
                pred_map[mid]["formation_energy_per_atom"]
                - gt_entry["formation_energy_per_atom"]
            ))
            bg_errors.append(abs(
                pred_map[mid]["band_gap"] - gt_entry["band_gap"]
            ))
        else:
            # Penalize missing predictions with a large error
            ef_errors.append(9999.0)
            bg_errors.append(9999.0)

    mae_ef = sum(ef_errors) / len(ef_errors) if ef_errors else 9999.0
    mae_bg = sum(bg_errors) / len(bg_errors) if bg_errors else 9999.0

    score_ef = normalize_score(mae_ef, BASELINE_MAE_EF)
    score_bg = normalize_score(mae_bg, BASELINE_MAE_BG)
    performance_score = score_ef + score_bg  # 0–40

    # ── Write results ─────────────────────────────────────────────────────────
    results = {
        "mae_formation_energy":  round(mae_ef,           4),
        "mae_band_gap":          round(mae_bg,           4),
        "score_formation_energy": round(score_ef,        2),
        "score_band_gap":         round(score_bg,        2),
        "performance_score":      round(performance_score, 2),
        "n_evaluated":            len(ef_errors) - len(missing),
        "n_missing":              len(missing),
    }

    out_path = f"{output_dir}/scores.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
