"""Bootstrap 95% CIs for the main PAS results.

Outputs:
    experiments/results/paper_tables/bootstrap_main.json
    experiments/results/paper_tables/bootstrap_main.md
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from experiments import config
from experiments.eval import metrics

CONFIGS = {
    "llm_ft": [
        "qwen3_t_seed42_test_r_generate",
        "qwen3_t_seed43_test_r_generate",
        "qwen3_t_seed44_test_r_generate",
    ],
    "llm_base": ["qwen3_base_test_r_generate"],
    "histgb": ["tabular_histgb_test_r"],
}
IN_DIST = [2016, 2018, 2019]


def _ci(xs: list[float], alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.percentile(xs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _bootstrap(df: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(df)
    maes, corrs, choices = [], [], []
    groups = df.groupby(["respondent_id", "year"])
    group_keys = list(groups.groups.keys())
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b = df.iloc[idx]
        maes.append(metrics.mae(b["true_score"], b["pred_float"]))
        if b["pred_float"].std() > 1e-9 and b["true_score"].std() > 1e-9:
            corrs.append(float(np.corrcoef(b["true_score"], b["pred_float"])[0, 1]))
        # Choice: resample respondent-year groups.
        keys = [group_keys[i] for i in rng.integers(0, len(group_keys), len(group_keys))]
        parts = [groups.get_group(k) for k in keys]
        bb = pd.concat(parts, ignore_index=True)
        choices.append(metrics.choice_accuracy(bb)["accuracy"])
    out = {}
    for name, vals in [("mae", maes), ("corr", corrs), ("choice", choices)]:
        vals = [v for v in vals if v == v]
        out[name] = {"mean": float(np.mean(vals)), "lo": _ci(vals)[0], "hi": _ci(vals)[1]}
    return out


def main() -> None:
    table = {}
    for model, stems in CONFIGS.items():
        frames = []
        for stem in stems:
            path = config.PREDS / f"{stem}.predictions.jsonl"
            if path.exists():
                frames.append(pd.read_json(path, lines=True))
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        table[model] = {
            "in_dist": _bootstrap(df[df["year"].isin(IN_DIST)]),
            "all": _bootstrap(df),
        }

    out_dir = config.RESULTS / "paper_tables"
    with open(out_dir / "bootstrap_main.json", "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    lines = ["# Bootstrap 95% CIs", "",
             "| model | split | MAE [95% CI] | corr [95% CI] | choice [95% CI] |",
             "|---|---|---|---|---|"]
    for model, entry in table.items():
        for split, r in entry.items():
            lines.append(
                f"| {model} | {split} | {r['mae']['mean']:.2f} "
                f"[{r['mae']['lo']:.2f}, {r['mae']['hi']:.2f}] | "
                f"{r['corr']['mean']:+.3f} [{r['corr']['lo']:+.3f}, {r['corr']['hi']:+.3f}] | "
                f"{r['choice']['mean']:.3f} [{r['choice']['lo']:.3f}, {r['choice']['hi']:.3f}] |"
            )
    (out_dir / "bootstrap_main.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
