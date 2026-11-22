"""Error profile by true-score quartile, with cluster bootstrap and permutation tests.

The profile table in the thesis reports, for each in-distribution wave, the
LLM and HistGB MAE in each true-score quartile. This script reproduces those
cells and adds the evidence the table previously lacked:

* respondent-cluster bootstrap 95% CIs for every cell, for the LLM-minus-HistGB
  difference, and for the U-shape contrast;
* a paired permutation test for each cell difference (swap the two models'
  predictions within a row), with Holm correction across the 12 cells;
* a permutation test for the U-shape contrast
  ``mean(delta in Q2,Q3) - mean(delta in Q1,Q4)``, where delta is LLM - HistGB;
* a permutation test for the 12/12 sign pattern (LLM better in Q1/Q4, worse in
  Q2/Q3 in every year);
* cluster-bootstrap CIs and a permutation test for the error--extremity
  correlation difference.

Rows are paired by (respondent_id, year, candidate). The LLM side is the mean
over seeds 42--44; the permutation swaps model labels for all seeds jointly, so
seed-to-seed correlation is preserved.

Outputs:
    experiments/results/diagnostics/error_profile.json
    experiments/results/diagnostics/error_profile.md
    experiments/results/diagnostics/error_profile_cells.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import config

KEY = ["respondent_id", "year", "candidate"]
YEARS = [2016, 2018, 2019]
SEEDS = [42, 43, 44]
QUARTILES = [1, 2, 3, 4]


def _load_llm(seed: int) -> pd.DataFrame:
    path = config.PREDS / f"qwen3_r_seed{seed}_test_r_generate.predictions.jsonl"
    df = pd.read_json(path, lines=True)
    df = df[df["year"].isin(YEARS)][KEY + ["true_score", "pred_float"]].copy()
    return df.rename(columns={"pred_float": f"pl{seed}"})


def _load_gbm() -> pd.DataFrame:
    path = config.PREDS / "tabular_histgb_test_r_r.predictions.jsonl"
    df = pd.read_json(path, lines=True)
    df = df[df["year"].isin(YEARS)][KEY + ["true_score", "pred_float"]].copy()
    return df.rename(columns={"pred_float": "pg"})


def _build_frame() -> pd.DataFrame:
    gbm = _load_gbm()
    df = gbm
    for seed in SEEDS:
        llm = _load_llm(seed)
        before = len(df)
        df = df.merge(llm, on=KEY, how="inner", suffixes=("", f"_{seed}"))
        if len(df) != before:
            raise RuntimeError(f"merge with seed {seed} changed row count")

    # The true score must be identical across seeds and the tabular file.
    true_cols = [c for c in df.columns if c.startswith("true_score")]
    for col in true_cols[1:]:
        if not np.allclose(df[true_cols[0]], df[col]):
            raise RuntimeError(f"true_score mismatch in {col}")
    df = df.drop(columns=true_cols[1:])
    df["quartile"] = (
        df.groupby("year", group_keys=False)["true_score"]
        .apply(lambda s: pd.qcut(s, 4, labels=False, duplicates="drop") + 1)
        .astype(int)
    )
    if sorted(df["quartile"].unique()) != QUARTILES:
        raise RuntimeError("quartile construction did not produce four bins")
    df["extremity"] = np.abs(df["true_score"] - df["true_score"].mean())
    return df.reset_index(drop=True)


def _cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in df.groupby("year"):
        for q, h in g.groupby("quartile"):
            llm_maes = [float(np.mean(np.abs(h["true_score"] - h[f"pl{s}"]))) for s in SEEDS]
            gbm_mae = float(np.mean(np.abs(h["true_score"] - h["pg"])))
            rows.append({
                "year": int(year),
                "quartile": int(q),
                "n_rows": int(len(h)),
                "n_respondents": int(h["respondent_id"].nunique()),
                "true_min": float(h["true_score"].min()),
                "true_max": float(h["true_score"].max()),
                "llm_mae": float(np.mean(llm_maes)),
                "llm_mae_seed_sd": float(np.std(llm_maes)),
                "gbm_mae": gbm_mae,
                "delta_mae": float(np.mean(llm_maes) - gbm_mae),
            })
    return pd.DataFrame(rows)


def _respondent_sums(df: pd.DataFrame) -> dict:
    """Precompute per-respondent sufficient statistics for fast cluster bootstrap.

    MAE cells use per-respondent sums of absolute errors and counts. The
    error--extremity correlation uses sums of x, x^2, y, y^2, x*y, and n.
    """
    out = {}
    for year, g in df.groupby("year"):
        codes, uniques = pd.factorize(g["respondent_id"], sort=True)
        n_resp = len(uniques)
        n_q = len(QUARTILES)

        # shape (n_resp, 4, n_seeds)
        sum_err_llm = np.zeros((n_resp, n_q, len(SEEDS)))
        sum_err_gbm = np.zeros((n_resp, n_q))
        count_q = np.zeros((n_resp, n_q))
        for qi, q in enumerate(QUARTILES):
            mask = g["quartile"].to_numpy() == q
            c = codes[mask]
            count_q[:, qi] = np.bincount(c, minlength=n_resp)
            sum_err_gbm[:, qi] = np.bincount(
                c, weights=np.abs(g.loc[mask, "true_score"] - g.loc[mask, "pg"]),
                minlength=n_resp)
            for si, s in enumerate(SEEDS):
                sum_err_llm[:, qi, si] = np.bincount(
                    c, weights=np.abs(g.loc[mask, "true_score"] - g.loc[mask, f"pl{s}"]),
                    minlength=n_resp)

        y = g["extremity"].to_numpy(float)
        stats = {
            "n_resp": n_resp,
            "n_rows": len(g),
            "sum_err_llm": sum_err_llm,
            "sum_err_gbm": sum_err_gbm,
            "count_q": count_q,
            "n": np.bincount(codes, minlength=n_resp).astype(float),
            "sum_y": np.bincount(codes, weights=y, minlength=n_resp),
            "sum_y2": np.bincount(codes, weights=y * y, minlength=n_resp),
            "sum_x": np.zeros((n_resp, len(SEEDS))),
            "sum_x2": np.zeros((n_resp, len(SEEDS))),
            "sum_xy": np.zeros((n_resp, len(SEEDS))),
        }
        for si, s in enumerate(SEEDS):
            x = np.abs(g["true_score"] - g[f"pl{s}"]).to_numpy(float)
            stats["sum_x"][:, si] = np.bincount(codes, weights=x, minlength=n_resp)
            stats["sum_x2"][:, si] = np.bincount(codes, weights=x * x, minlength=n_resp)
            stats["sum_xy"][:, si] = np.bincount(codes, weights=x * y, minlength=n_resp)
        x = np.abs(g["true_score"] - g["pg"]).to_numpy(float)
        stats["sum_x_gbm"] = np.bincount(codes, weights=x, minlength=n_resp)
        stats["sum_x2_gbm"] = np.bincount(codes, weights=x * x, minlength=n_resp)
        stats["sum_xy_gbm"] = np.bincount(codes, weights=x * y, minlength=n_resp)
        out[int(year)] = stats
    return out


def _corr_from_sums(sum_x, sum_x2, sum_xy, n, sum_y, sum_y2):
    """Pearson correlation from weighted sums (arrays or scalars)."""
    mx = sum_x / n
    my = sum_y / n
    cov = sum_xy / n - mx * my
    vx = sum_x2 / n - mx * mx
    vy = sum_y2 / n - my * my
    return cov / np.sqrt(np.maximum(vx * vy, 1e-24))


def _bootstrap(stats_by_year: dict, n_boot: int, seed: int) -> dict:
    """Respondent-cluster bootstrap for all cells and global correlations."""
    rng = np.random.default_rng(seed)
    years = sorted(stats_by_year)
    n_q = len(QUARTILES)
    n_s = len(SEEDS)

    boot = {
        "llm_mae": {y: np.zeros((n_boot, n_q)) for y in years},
        "gbm_mae": {y: np.zeros((n_boot, n_q)) for y in years},
        "delta_mae": {y: np.zeros((n_boot, n_q)) for y in years},
    }
    # Accumulate weighted sufficient statistics across years so that the
    # correlation is the global one reported in the thesis, not an average of
    # per-year correlations.
    g_n = np.zeros(n_boot)
    g_sy = np.zeros(n_boot)
    g_sy2 = np.zeros(n_boot)
    g_sx = np.zeros((n_boot, n_s))
    g_sx2 = np.zeros((n_boot, n_s))
    g_sxy = np.zeros((n_boot, n_s))
    g_sx_gbm = np.zeros(n_boot)
    g_sx2_gbm = np.zeros(n_boot)
    g_sxy_gbm = np.zeros(n_boot)

    for year in years:
        st = stats_by_year[year]
        n_resp = st["n_resp"]
        # One multinomial draw per bootstrap replicate gives cluster weights.
        weights = rng.multinomial(n_resp, np.ones(n_resp) / n_resp, size=n_boot)
        # shape (n_boot, 4)
        denom_q = weights @ st["count_q"]
        llm_num = np.einsum("br,rqs->bqs", weights, st["sum_err_llm"])
        gbm_num = weights @ st["sum_err_gbm"]
        llm_mae = llm_num / denom_q[:, :, None]
        gbm_mae = gbm_num / denom_q
        boot["llm_mae"][year] = llm_mae.mean(axis=2)
        boot["gbm_mae"][year] = gbm_mae
        boot["delta_mae"][year] = llm_mae.mean(axis=2) - gbm_mae

        g_n += weights @ st["n"]
        g_sy += weights @ st["sum_y"]
        g_sy2 += weights @ st["sum_y2"]
        for si in range(n_s):
            g_sx[:, si] += weights @ st["sum_x"][:, si]
            g_sx2[:, si] += weights @ st["sum_x2"][:, si]
            g_sxy[:, si] += weights @ st["sum_xy"][:, si]
        g_sx_gbm += weights @ st["sum_x_gbm"]
        g_sx2_gbm += weights @ st["sum_x2_gbm"]
        g_sxy_gbm += weights @ st["sum_xy_gbm"]

    corr_llm = np.column_stack([
        _corr_from_sums(g_sx[:, si], g_sx2[:, si], g_sxy[:, si],
                        g_n, g_sy, g_sy2)
        for si in range(n_s)
    ]).mean(axis=1)
    corr_gbm = _corr_from_sums(g_sx_gbm, g_sx2_gbm, g_sxy_gbm,
                               g_n, g_sy, g_sy2)
    boot["corr_llm"] = corr_llm
    boot["corr_gbm"] = corr_gbm
    boot["corr_diff"] = corr_llm - corr_gbm
    return boot


def _permutation(df: pd.DataFrame, n_perm: int, seed: int,
                 chunk: int = 2000) -> dict:
    """Paired permutation test swapping model labels within a row.

    For each permutation, every row independently swaps the LLM and HistGB
    predictions; all seeds swap together. Per-quartile deltas are stored per
    year; the error--extremity correlation is computed globally over all
    in-distribution rows.
    """
    rng = np.random.default_rng(seed)
    perms = []
    for year, g in df.groupby("year"):
        n = len(g)
        q = g["quartile"].to_numpy()
        y = g["true_score"].to_numpy(float)
        ext = g["extremity"].to_numpy(float)
        A = np.column_stack([np.abs(y - g[f"pl{s}"].to_numpy(float)) for s in SEEDS])
        B = np.abs(y - g["pg"].to_numpy(float))
        D = A - B[:, None]  # (n, n_seeds)
        q_masks = [q == qi for qi in QUARTILES]
        n_q = np.array([m.sum() for m in q_masks], float)
        obs_delta = np.array([
            np.mean(D[m].mean(axis=1)) for m in q_masks
        ])  # mean over seeds of mean over rows
        perms.append({
            "year": int(year),
            "n": n,
            "true": y,
            "ext": ext,
            "A": A,
            "B": B,
            "D": D,
            "q_masks": q_masks,
            "n_q": n_q,
            "obs_delta": obs_delta,
        })

    # Observed global correlation across all in-distribution rows.
    A_all = np.vstack([p["A"] for p in perms])
    B_all = np.concatenate([p["B"] for p in perms])
    ext_all = np.concatenate([p["ext"] for p in perms])
    obs_corr_llm = float(np.mean([
        np.corrcoef(A_all[:, si], ext_all)[0, 1] for si in range(len(SEEDS))
    ]))
    obs_corr_gbm = float(np.corrcoef(B_all, ext_all)[0, 1])

    for p in perms:
        p["delta_perm"] = np.zeros((n_perm, len(QUARTILES)))
    corr_diff_perm = np.zeros(n_perm)

    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm - start)
        # Global sufficient statistics for the correlation under this chunk.
        sum_e = np.zeros((m, len(SEEDS)))
        sum_e2 = np.zeros((m, len(SEEDS)))
        sum_eext = np.zeros((m, len(SEEDS)))
        sum_g = np.zeros((m, len(SEEDS)))
        sum_g2 = np.zeros((m, len(SEEDS)))
        sum_gext = np.zeros((m, len(SEEDS)))
        n_total = 0
        sum_ext = 0.0
        sum_ext2 = 0.0

        for p in perms:
            n = p["n"]
            # +1 keeps the labels, -1 swaps them.
            Z = rng.choice(np.array([-1.0, 1.0]), size=(m, n))
            # delta_q = mean over rows in q of mean over seeds of Z*D
            for qi, mask in enumerate(p["q_masks"]):
                p["delta_perm"][start:start + m, qi] = (
                    Z[:, mask] @ p["D"][mask].mean(axis=1) / p["n_q"][qi]
                )
            for si in range(len(SEEDS)):
                a = p["A"][:, si]
                b = p["B"]
                e_llm = a + (1 - Z) / 2 * (b - a)
                e_gbm = b + (1 - Z) / 2 * (a - b)
                sum_e[:, si] += e_llm.sum(axis=1)
                sum_e2[:, si] += (e_llm * e_llm).sum(axis=1)
                sum_eext[:, si] += e_llm @ p["ext"]
                sum_g[:, si] += e_gbm.sum(axis=1)
                sum_g2[:, si] += (e_gbm * e_gbm).sum(axis=1)
                sum_gext[:, si] += e_gbm @ p["ext"]
            n_total += n
            sum_ext += p["ext"].sum()
            sum_ext2 += (p["ext"] ** 2).sum()

        corr_llm = np.column_stack([
            _corr_from_sums(sum_e[:, si], sum_e2[:, si], sum_eext[:, si],
                            n_total, sum_ext, sum_ext2)
            for si in range(len(SEEDS))
        ]).mean(axis=1)
        corr_gbm = np.column_stack([
            _corr_from_sums(sum_g[:, si], sum_g2[:, si], sum_gext[:, si],
                            n_total, sum_ext, sum_ext2)
            for si in range(len(SEEDS))
        ]).mean(axis=1)
        corr_diff_perm[start:start + m] = corr_llm - corr_gbm

    return {
        "years": perms,
        "obs_corr_llm": obs_corr_llm,
        "obs_corr_gbm": obs_corr_gbm,
        "corr_diff_perm": corr_diff_perm,
    }


def _p_two_sided(obs: float, null: np.ndarray) -> float:
    return float((1 + np.sum(np.abs(null) >= abs(obs))) / (len(null) + 1))


def _p_greater(obs: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= obs)) / (len(null) + 1))


def _fmt_p(p: float) -> str:
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def _holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return [float(v) for v in adj]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = _build_frame()
    cells = _cell_stats(df)
    stats_by_year = _respondent_sums(df)
    boot = _bootstrap(stats_by_year, args.bootstrap, args.seed)
    perm = _permutation(df, args.permutations, args.seed + 1)
    perm_by_year = {p["year"]: p for p in perm["years"]}

    # Attach CIs and permutation p-values to every cell.
    records = []
    pvals = []
    for _, row in cells.iterrows():
        year, q = int(row["year"]), int(row["quartile"])
        qi = q - 1
        llm_ci = np.percentile(boot["llm_mae"][year][:, qi], [2.5, 97.5])
        gbm_ci = np.percentile(boot["gbm_mae"][year][:, qi], [2.5, 97.5])
        d_ci = np.percentile(boot["delta_mae"][year][:, qi], [2.5, 97.5])
        null = perm_by_year[year]["delta_perm"][:, qi]
        p = _p_two_sided(row["delta_mae"], null)
        pvals.append(p)
        records.append({
            **row.to_dict(),
            "year": int(year),
            "quartile": int(q),
            "n_rows": int(row["n_rows"]),
            "n_respondents": int(row["n_respondents"]),
            "llm_mae_ci": [float(llm_ci[0]), float(llm_ci[1])],
            "gbm_mae_ci": [float(gbm_ci[0]), float(gbm_ci[1])],
            "delta_mae_ci": [float(d_ci[0]), float(d_ci[1])],
            "delta_p": p,
        })
    holm = _holm(pvals)
    for rec, hp in zip(records, holm):
        rec["delta_p_holm"] = hp

    # U-shape contrast per permutation: middle minus tails.
    tail_idx = [0, 3]
    mid_idx = [1, 2]
    T_perm = np.zeros(args.permutations)
    for py in perm["years"]:
        d = py["delta_perm"]
        T_perm += (d[:, mid_idx].mean(axis=1) - d[:, tail_idx].mean(axis=1)) / len(perm["years"])
    T_obs = float(np.mean([
        (py["obs_delta"][mid_idx].mean() - py["obs_delta"][tail_idx].mean())
        for py in perm["years"]
    ]))
    t_boot = []
    for b in range(args.bootstrap):
        per_year = []
        for year in YEARS:
            d = boot["delta_mae"][year][b]
            per_year.append(d[mid_idx].mean() - d[tail_idx].mean())
        t_boot.append(np.mean(per_year))
    t_ci = np.percentile(t_boot, [2.5, 97.5])
    t_p = _p_greater(T_obs, T_perm)

    # 12-cell sign pattern.
    expected = {-1: [1, 4], 1: [2, 3]}  # sign of delta -> quartiles where it should hold
    obs_matches = 0
    for r in records:
        sign = np.sign(r["delta_mae"])
        obs_matches += int(r["quartile"] in expected.get(sign, []))
    match_perm = np.zeros(args.permutations)
    for py in perm["years"]:
        d = py["delta_perm"]
        match_perm += (
            (d[:, 0] < 0).astype(int) + (d[:, 1] > 0).astype(int) +
            (d[:, 2] > 0).astype(int) + (d[:, 3] < 0).astype(int)
        )
    pattern_p = _p_greater(obs_matches, match_perm)

    # Correlation summary.
    corr_llm = perm["obs_corr_llm"]
    corr_gbm = perm["obs_corr_gbm"]
    corr_diff = corr_llm - corr_gbm
    corr_llm_ci = np.percentile(boot["corr_llm"], [2.5, 97.5])
    corr_gbm_ci = np.percentile(boot["corr_gbm"], [2.5, 97.5])
    corr_diff_ci = np.percentile(boot["corr_diff"], [2.5, 97.5])
    corr_null = perm["corr_diff_perm"]
    corr_p = _p_two_sided(corr_diff, corr_null)

    out = {
        "years": YEARS,
        "seeds": SEEDS,
        "n_rows": int(len(df)),
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "cells": records,
        "u_shape": {
            "definition": "mean(delta Q2,Q3) - mean(delta Q1,Q4); positive = LLM relatively worse in the middle",
            "estimate": T_obs,
            "ci": [float(t_ci[0]), float(t_ci[1])],
            "p_one_sided": t_p,
        },
        "pattern": {
            "observed_matches": int(obs_matches),
            "total_cells": len(records),
            "p": pattern_p,
        },
        "error_extremity": {
            "definition": "Pearson corr(|error|, |true_score - mean(true_score)|) over all in-distribution rows; LLM is the mean over seeds",
            "llm": corr_llm,
            "llm_ci": [float(corr_llm_ci[0]), float(corr_llm_ci[1])],
            "gbm": corr_gbm,
            "gbm_ci": [float(corr_gbm_ci[0]), float(corr_gbm_ci[1])],
            "difference": corr_diff,
            "difference_ci": [float(corr_diff_ci[0]), float(corr_diff_ci[1])],
            "difference_p": corr_p,
        },
    }

    out_dir = config.RESULTS / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "error_profile.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.DataFrame(records).to_csv(out_dir / "error_profile_cells.csv", index=False)

    lines = [
        "# Error profile by true-score quartile",
        "",
        f"Rows: {len(df):,}; years: {', '.join(map(str, YEARS))}; "
        f"LLM mean over seeds {', '.join(map(str, SEEDS))}; HistGB single run.",
        f"Bootstrap: {args.bootstrap:,} respondent-cluster resamples; "
        f"permutations: {args.permutations:,}.",
        "",
        "| Year | Q | n | LLM MAE [95% CI] | HistGB MAE [95% CI] | Delta (LLM-GBM) [95% CI] | p | p (Holm) |",
        "|---|---:|---:|---|---|---|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['year']} | Q{r['quartile']} | {r['n_rows']} | "
            f"{r['llm_mae']:.2f} [{r['llm_mae_ci'][0]:.2f}, {r['llm_mae_ci'][1]:.2f}] | "
            f"{r['gbm_mae']:.2f} [{r['gbm_mae_ci'][0]:.2f}, {r['gbm_mae_ci'][1]:.2f}] | "
            f"{r['delta_mae']:+.2f} [{r['delta_mae_ci'][0]:+.2f}, {r['delta_mae_ci'][1]:+.2f}] | "
            f"{_fmt_p(r['delta_p'])} | {_fmt_p(r['delta_p_holm'])} |"
        )
    lines += [
        "",
        "## Global tests",
        "",
        f"- U-shape contrast (middle minus tails): **{T_obs:+.2f}** "
        f"[{t_ci[0]:+.2f}, {t_ci[1]:+.2f}], one-sided p = **{_fmt_p(t_p)}**.",
        f"- Sign pattern: **{obs_matches}/{len(records)}** cells match "
        f"(LLM better in Q1/Q4, worse in Q2/Q3); p = **{_fmt_p(pattern_p)}**.",
        f"- Error-extremity correlation: LLM **{corr_llm:+.3f}** "
        f"[{corr_llm_ci[0]:+.3f}, {corr_llm_ci[1]:+.3f}], HistGB **{corr_gbm:+.3f}** "
        f"[{corr_gbm_ci[0]:+.3f}, {corr_gbm_ci[1]:+.3f}], difference "
        f"**{corr_diff:+.3f}** [{corr_diff_ci[0]:+.3f}, {corr_diff_ci[1]:+.3f}], "
        f"p = **{_fmt_p(corr_p)}**.",
        "",
        "Delta is LLM minus HistGB, so negative means the LLM has lower MAE.",
    ]
    (out_dir / "error_profile.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
