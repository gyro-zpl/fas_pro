"""Tables for the respondent-disjoint protocol (train_r -> test_r).

Reads:
    experiments/results/preds/qwen3_r_seed{42,43,44}_test_r_generate.predictions.jsonl
    experiments/results/preds/tabular_histgb_test_r_r.predictions.jsonl
    experiments/results/preds/tabular_ridge_test_r_r.predictions.jsonl
    experiments/results/preds/qwen3_base_test_r_generate.predictions.jsonl
    experiments/results/preds/deepseek_r_seed42_test_r_generate.predictions.jsonl
Writes:
    experiments/results/paper_tables/r_protocol.md / .json
    experiments/results/paper_tables/r_pas_by_year.md
    experiments/results/paper_tables/r_cross_pressure.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments import config
from experiments.eval import metrics

IN_DIST = [2016, 2018, 2019]
KEY = ["respondent_id", "year", "candidate"]
SEEDS = [42, 43, 44]


def _load(stem: str) -> pd.DataFrame:
    df = pd.read_json(config.PREDS / f"{stem}.predictions.jsonl", lines=True)
    return df[df["year"].isin(IN_DIST)].copy()


def _load_all(stem: str) -> pd.DataFrame:
    return pd.read_json(config.PREDS / f"{stem}.predictions.jsonl", lines=True)


def _corr(df: pd.DataFrame) -> float:
    if df["pred_float"].std() < 1e-9 or df["true_score"].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(df["true_score"], df["pred_float"])[0, 1])


def _choice_scores(df: pd.DataFrame) -> tuple[float, float, int]:
    """Return (idxmax accuracy, tie-aware accuracy, n_groups)."""
    exact = 0.0
    aware = 0.0
    groups = 0
    for _, g in df.groupby(["respondent_id", "year"]):
        if len(g) < 2:
            continue
        groups += 1
        ts = g["true_score"].to_numpy()
        ps = g["pred_float"].to_numpy()
        true_set = set(g.index[ts == ts.max()])
        pred_set = set(g.index[ps == ps.max()])
        exact += int(g.loc[g["pred_float"].idxmax()].name ==
                     g.loc[g["true_score"].idxmax()].name)
        aware += len(true_set & pred_set) / len(pred_set)
    if groups == 0:
        return float("nan"), float("nan"), 0
    return exact / groups, aware / groups, groups


def _metrics(df: pd.DataFrame) -> dict:
    exact, aware, groups = _choice_scores(df)
    return {
        "n": int(len(df)),
        "mae": metrics.mae(df["true_score"], df["pred_float"]),
        "corr": _corr(df),
        "choice": exact,
        "choice_tie_aware": aware,
        "n_groups": groups,
        "pred_std": float(df["pred_float"].std()),
        "chance": float(metrics.choice_accuracy(df)["chance"]),
    }


def _mean_std(vals: list[float]) -> tuple[float, float]:
    xs = [v for v in vals if v == v]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), float("nan"))


def _bootstrap_rows(df: pd.DataFrame, seed: int = 42, n_boot: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    n = len(df)
    maes, corrs = [], []
    for _ in range(n_boot):
        b = df.iloc[rng.integers(0, n, n)]
        maes.append(metrics.mae(b["true_score"], b["pred_float"]))
        c = _corr(b)
        if c == c:
            corrs.append(c)
    return {
        "mae": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
        "corr": [float(np.percentile(corrs, 2.5)), float(np.percentile(corrs, 97.5))],
    }


def _bootstrap_clusters(df: pd.DataFrame, seed: int = 42, n_boot: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    resp = df["respondent_id"].unique()
    groups = {r: g for r, g in df.groupby("respondent_id")}
    maes, corrs = [], []
    for _ in range(n_boot):
        rs = rng.choice(resp, len(resp), replace=True)
        b = pd.concat([groups[r] for r in rs], ignore_index=True)
        maes.append(metrics.mae(b["true_score"], b["pred_float"]))
        c = _corr(b)
        if c == c:
            corrs.append(c)
    return {
        "mae": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
        "corr": [float(np.percentile(corrs, 2.5)), float(np.percentile(corrs, 97.5))],
    }


def _bootstrap_choice(df: pd.DataFrame, seed: int = 42, n_boot: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for (r, y), g in df.groupby(["respondent_id", "year"]):
        if len(g) < 2:
            continue
        ts = g["true_score"].to_numpy()
        ps = g["pred_float"].to_numpy()
        true_set = set(g.index[ts == ts.max()])
        pred_set = set(g.index[ps == ps.max()])
        rows.append((int(g.loc[g["pred_float"].idxmax()].name ==
                         g.loc[g["true_score"].idxmax()].name),
                     len(true_set & pred_set) / len(pred_set)))
    arr = np.asarray(rows, float)
    exact, aware = [], []
    for _ in range(n_boot):
        b = arr[rng.integers(0, len(arr), len(arr))]
        exact.append(b[:, 0].mean())
        aware.append(b[:, 1].mean())
    return {
        "choice": [float(np.percentile(exact, 2.5)), float(np.percentile(exact, 97.5))],
        "choice_tie_aware": [float(np.percentile(aware, 2.5)), float(np.percentile(aware, 97.5))],
    }


def _paired(llm: pd.DataFrame, gbm: pd.DataFrame, seed: int = 42,
            n_boot: int = 1000) -> dict:
    b = gbm.drop(columns=["true_score"]).rename(
        columns={"pred": "pred_b", "pred_float": "pred_float_b"})
    a = llm[KEY + ["true_score", "pred_float"]].rename(columns={"pred_float": "pred_a"})
    df = a.merge(b, on=KEY, how="inner").reset_index(drop=True)
    y = df["true_score"].to_numpy(float)
    pa = df["pred_a"].to_numpy(float)
    pb = df["pred_float_b"].to_numpy(float)
    err_a, err_b = np.abs(y - pa), np.abs(y - pb)

    group_rows = []
    for _, g in df.groupby(["respondent_id", "year"]):
        if len(g) < 2:
            continue
        ts = g["true_score"].to_numpy()
        true_set = set(g.index[ts == ts.max()])
        pa_set = set(g.index[g["pred_a"].to_numpy() == g["pred_a"].max()])
        pb_set = set(g.index[g["pred_float_b"].to_numpy() == g["pred_float_b"].max()])
        group_rows.append((
            int(g.loc[g["pred_a"].idxmax()].name == g.loc[g["true_score"].idxmax()].name),
            int(g.loc[g["pred_float_b"].idxmax()].name == g.loc[g["true_score"].idxmax()].name),
            len(true_set & pa_set) / len(pa_set),
            len(true_set & pb_set) / len(pb_set),
        ))
    groups = np.asarray(group_rows, float)

    rng = np.random.default_rng(seed)
    n = len(df)
    d_mae, d_corr, d_choice, d_aware = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d_mae.append(float(err_a[idx].mean() - err_b[idx].mean()))
        yy, aa, bb = y[idx], pa[idx], pb[idx]
        if aa.std() > 1e-9 and bb.std() > 1e-9:
            d_corr.append(float(np.corrcoef(yy, aa)[0, 1] - np.corrcoef(yy, bb)[0, 1]))
        gi = rng.integers(0, len(groups), len(groups))
        d_choice.append(float(groups[gi, 0].mean() - groups[gi, 1].mean()))
        d_aware.append(float(groups[gi, 2].mean() - groups[gi, 3].mean()))

    def ci(vals):
        vals = [v for v in vals if v == v]
        return [float(np.mean(vals)), float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5))]

    return {
        "n": int(n),
        "delta_mae": ci(d_mae),
        "delta_corr": ci(d_corr),
        "delta_choice": ci(d_choice),
        "delta_choice_tie_aware": ci(d_aware),
    }


def _per_year(df: pd.DataFrame) -> dict:
    out = {}
    for year, g in df.groupby("year"):
        m = _metrics(g)
        m["chance"] = float(metrics.choice_accuracy(g)["chance"])
        out[int(year)] = m
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44",
                    help="comma-separated Qwen seeds to include")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s]
    llm = {}
    for s in seeds:
        path = config.PREDS / f"qwen3_r_seed{s}_test_r_generate.predictions.jsonl"
        if path.exists():
            llm[s] = _load(f"qwen3_r_seed{s}_test_r_generate")
        else:
            print(f"missing {path}; skipping seed {s}")
    if not llm:
        raise SystemExit("no Qwen r predictions found")
    seeds = sorted(llm)
    gbm = _load("tabular_histgb_test_r_r")
    ridge = _load("tabular_ridge_test_r_r")
    base = _load("qwen3_base_test_r_generate")
    deepseek_path = config.PREDS / "deepseek_r_seed42_test_r_generate.predictions.jsonl"
    deepseek = _load("deepseek_r_seed42_test_r_generate") if deepseek_path.exists() else None

    table = {}
    for name, df in [("llm_r", llm), ("histgb_r", gbm), ("ridge_r", ridge),
                     ("base", base)] + ([("deepseek_r", deepseek)] if deepseek is not None else []):
        if isinstance(df, dict):
            per_seed = {str(s): _metrics(d) for s, d in df.items()}
            entry = {"per_seed": per_seed}
            for key in ["mae", "corr", "choice", "choice_tie_aware", "pred_std"]:
                entry[f"{key}_mean"], entry[f"{key}_std"] = _mean_std(
                    [v[key] for v in per_seed.values()])
            entry["n"] = int(sum(v["n"] for v in per_seed.values()) / len(per_seed))
            entry["n_groups"] = int(per_seed[str(seeds[0])]["n_groups"])
            entry["chance"] = float(per_seed[str(seeds[0])]["chance"])
            # CIs: pooled rows for MAE/corr; per-seed group bootstrap averaged
            pooled = pd.concat(df.values(), ignore_index=True)
            entry["ci"] = _bootstrap_rows(pooled)
            entry["ci_cluster"] = _bootstrap_clusters(pooled)
            choice_cis = [_bootstrap_choice(d) for d in df.values()]
            entry["ci"]["choice"] = [
                float(np.mean([c["choice"][0] for c in choice_cis])),
                float(np.mean([c["choice"][1] for c in choice_cis])),
            ]
            entry["ci"]["choice_tie_aware"] = [
                float(np.mean([c["choice_tie_aware"][0] for c in choice_cis])),
                float(np.mean([c["choice_tie_aware"][1] for c in choice_cis])),
            ]
            # Per-year metrics are averaged over seeds; n is the per-seed n.
            per_year = {}
            for year in sorted({int(y) for d in df.values() for y in d["year"].unique()}):
                vals = [_metrics(d[d["year"] == year]) for d in df.values()]
                rec = {key: float(np.mean([v[key] for v in vals]))
                       for key in ["mae", "corr", "choice", "choice_tie_aware",
                                   "pred_std", "chance"]}
                rec["n"] = int(vals[0]["n"])
                rec["n_groups"] = int(vals[0]["n_groups"])
                per_year[year] = rec
            entry["per_year"] = per_year
        else:
            entry = _metrics(df)
            entry["ci"] = _bootstrap_rows(df)
            entry["ci_cluster"] = _bootstrap_clusters(df)
            entry["ci"].update(_bootstrap_choice(df))
            entry["per_year"] = _per_year(df)
        table[name] = entry

    # Paired comparisons on the same rows.
    paired = {}
    for s, df in llm.items():
        paired[f"llm_seed{s}_minus_gbm"] = _paired(df, gbm)
        paired[f"llm_seed{s}_minus_base"] = _paired(df, base)
    paired["deepseek_minus_gbm"] = _paired(deepseek, gbm) if deepseek is not None else None

    # Cross-pressure subgroups.
    clean = pd.read_csv(config.V3 / "anes_candidates_clean.csv", low_memory=False)
    clean["pm"] = pd.to_numeric(clean["party_match"], errors="coerce")
    clean["idist"] = pd.to_numeric(clean["ideology_distance_7"], errors="coerce")
    clean["sim"] = pd.to_numeric(clean["demographic_similarity"], errors="coerce")
    clean["cp1"] = (clean["pm"] == 1) & (clean["idist"] >= 3)
    clean["cp2"] = (clean["pm"] == 0) & (clean["idist"] <= 1)
    clean["cp3"] = (clean["sim"] >= 0.7) & (clean["pm"] == 0)
    clean["cp_type"] = np.select([clean["cp1"], clean["cp2"], clean["cp3"]],
                                 ["CP1", "CP2", "CP3"], default="none")
    cp = clean[KEY + ["cp_type"]].drop_duplicates(KEY)
    cp_out = {}
    cp_configs = [(f"llm_seed{s}", llm[s]) for s in seeds] + [("histgb_r", gbm)]
    for label, df in cp_configs:
        d = df.merge(cp, on=KEY, how="left")
        entry = {}
        for cp_type, g in d.groupby("cp_type"):
            entry[cp_type] = {"n": int(len(g)),
                              "mae": metrics.mae(g["true_score"], g["pred_float"]),
                              "corr": _corr(g)}
        cp_out[label] = entry

    out_dir = config.RESULTS / "paper_tables"
    with open(out_dir / "r_protocol.json", "w") as f:
        json.dump({"table": table, "paired": paired, "cross_pressure": cp_out},
                  f, indent=2, ensure_ascii=False)

    def fmt(x, nd=4):
        return f"{x:.{nd}f}"

    lines = ["# Respondent-disjoint protocol (train_r -> test_r)", ""]
    lines.append("In-distribution rows: 2016, 2018, 2019.")
    lines.append("")
    lines.append("| Model | n | MAE | corr | choice | tie-aware | pred std |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ["llm_r", "histgb_r", "ridge_r", "base"] + (["deepseek_r"] if deepseek is not None else []):
        e = table[name]
        if "per_seed" in e:
            lines.append(
                f"| {name} | {e['n']} | {e['mae_mean']:.2f} ± {e['mae_std']:.2f} | "
                f"{e['corr_mean']:.3f} ± {e['corr_std']:.3f} | "
                f"{e['choice_mean']:.3f} ± {e['choice_std']:.3f} | "
                f"{e['choice_tie_aware_mean']:.3f} ± {e['choice_tie_aware_std']:.3f} | "
                f"{e['pred_std_mean']:.1f} |")
        else:
            lines.append(
                f"| {name} | {e['n']} | {e['mae']:.2f} | {e['corr']:.3f} | "
                f"{e['choice']:.3f} | {e['choice_tie_aware']:.3f} | {e['pred_std']:.1f} |")
    lines.append("")
    lines.append("| Model | MAE 95% CI (rows) | MAE 95% CI (clusters) | corr 95% CI (rows) | corr 95% CI (clusters) |")
    lines.append("|---|---|---|---|---|")
    for name in table:
        c, cc = table[name]["ci"], table[name]["ci_cluster"]
        lines.append(f"| {name} | [{c['mae'][0]:.2f}, {c['mae'][1]:.2f}] | "
                     f"[{cc['mae'][0]:.2f}, {cc['mae'][1]:.2f}] | "
                     f"[{c['corr'][0]:.3f}, {c['corr'][1]:.3f}] | "
                     f"[{cc['corr'][0]:.3f}, {cc['corr'][1]:.3f}] |")
    lines.append("")
    lines.append("## Paired bootstrap")
    lines.append("")
    lines.append("| comparison | n | ΔMAE | Δcorr | Δchoice | Δchoice (tie-aware) |")
    lines.append("|---|---|---|---|---|---|")
    for name, r in paired.items():
        if r is None:
            continue
        lines.append(
            f"| {name} | {r['n']} | {r['delta_mae'][0]:+.2f} "
            f"[{r['delta_mae'][1]:+.2f}, {r['delta_mae'][2]:+.2f}] | "
            f"{r['delta_corr'][0]:+.3f} [{r['delta_corr'][1]:+.3f}, {r['delta_corr'][2]:+.3f}] | "
            f"{r['delta_choice'][0]:+.3f} [{r['delta_choice'][1]:+.3f}, {r['delta_choice'][2]:+.3f}] | "
            f"{r['delta_choice_tie_aware'][0]:+.3f} [{r['delta_choice_tie_aware'][1]:+.3f}, "
            f"{r['delta_choice_tie_aware'][2]:+.3f}] |")
    (out_dir / "r_protocol.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # Per-year table (mean over Qwen seeds).
    years = sorted(table["llm_r"]["per_year"].keys())
    lines = ["# R protocol by year", "",
             "| year | n | LLM MAE | LLM corr | LLM choice | LLM tie-aware | GBM MAE | GBM corr | GBM choice |",
             "|---|---|---|---|---|---|---|---|---|"]
    for y in years:
        l = table["llm_r"]["per_year"][y]
        g = table["histgb_r"]["per_year"][y]
        lines.append(f"| {y} | {l['n']} | {l['mae']:.2f} | {l['corr']:+.3f} | "
                     f"{l['choice']:.3f} | {l['choice_tie_aware']:.3f} | "
                     f"{g['mae']:.2f} | {g['corr']:+.3f} | {g['choice']:.3f} |")
    (out_dir / "r_pas_by_year.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    lines = ["# Cross-pressure subgroups, R protocol (2016-2019)", "",
             "| config | CP1 | CP2 | CP3 | none |", "|---|---|---|---|---|"]
    for name, entry in cp_out.items():
        cells = []
        for t in ["CP1", "CP2", "CP3", "none"]:
            r = entry.get(t)
            cells.append(f"n={r['n']}, MAE={r['mae']:.2f}, r={r['corr']:+.3f}" if r else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (out_dir / "r_cross_pressure.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # 2024 test_t (all respondents) for the r models.
    test_t = {}
    test_t_paths = {f"qwen3_r_seed{s}": f"qwen3_r_seed{s}_test_t_generate" for s in seeds}
    if deepseek is not None:
        test_t_paths["deepseek_r"] = "deepseek_r_seed42_test_t_generate"
    for name, stem in test_t_paths.items():
        p = config.PREDS / f"{stem}.predictions.jsonl"
        if not p.exists():
            continue
        test_t[name] = _metrics(_load_all(stem))
    lines = ["# R protocol: 2024 test_t", "",
             "| model | n | MAE | corr | choice | tie-aware |",
             "|---|---|---|---|---|---|"]
    for name, e in test_t.items():
        lines.append(f"| {name} | {e['n']} | {e['mae']:.2f} | {e['corr']:+.3f} | "
                     f"{e['choice']:.3f} | {e['choice_tie_aware']:.3f} |")
    (out_dir / "r_test_t.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # Strict T x R cell: 226 rows from test_tr, filtered out of test_t predictions.
    tr_keys = set()
    with open(config.V3_SPLITS / "test_tr.jsonl") as f:
        for line in f:
            m = json.loads(line)["metadata"]
            tr_keys.add((int(m["respondent_id"]), int(m["year"]), str(m["candidate"])))
    tr_out = {}
    for name, stem in test_t_paths.items():
        p = config.PREDS / f"{stem}.predictions.jsonl"
        if not p.exists():
            continue
        df = _load_all(stem)
        mask = [(int(r), int(y), str(c)) in tr_keys
                for r, y, c in zip(df["respondent_id"], df["year"], df["candidate"])]
        d = df[mask]
        if len(d):
            tr_out[name] = _metrics(d)
    lines = ["# Strict T x R cell (2024, 113 respondents, 226 rows)", "",
             "| model | n | MAE | corr | choice | tie-aware |",
             "|---|---|---|---|---|---|"]
    for name, e in tr_out.items():
        lines.append(f"| {name} | {e['n']} | {e['mae']:.2f} | {e['corr']:+.3f} | "
                     f"{e['choice']:.3f} | {e['choice_tie_aware']:.3f} |")
    (out_dir / "r_test_tr.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
