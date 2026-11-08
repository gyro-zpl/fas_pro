"""Generate the paper figures (PDF + PNG).

Outputs:
    docs/04_论文稿件/figures/fig2_temporal_decay.{pdf,png}
    docs/04_论文稿件/figures/fig3_fas_js.{pdf,png}
    docs/04_论文稿件/figures/fig4_pas_scatter.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import config

# Validated palette (light mode, dataviz reference).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#8a8a85"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e0"

OUT = config.ROOT / "docs" / "04_论文稿件" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})


def _save(fig, name: str) -> None:
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def fig_temporal() -> None:
    table = json.load(open(config.RESULTS / "paper_tables" / "pas_by_year.json"))
    years = [2016, 2018, 2019, 2020, 2022, 2024]

    def series(model, key):
        vals, stds = [], []
        for y in years:
            r = table[f"{model}|test_r|{y}"]
            vals.append(r[key])
            stds.append(r.get(f"{key}_std", 0.0))
        return np.array(vals), np.array(stds)

    llm_corr, llm_corr_sd = series("llm_ft", "corr")
    gbm_corr, _ = series("histgb", "corr")
    llm_mae, llm_mae_sd = series("llm_ft", "mae")
    gbm_mae, _ = series("histgb", "mae")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    x = np.arange(len(years))

    ax = axes[0]
    ax.axhline(0, color=MUTED, linewidth=0.7, linestyle=":")
    ax.plot(x, llm_corr, color=BLUE, linewidth=2, marker="o", markersize=5,
            label="Fine-tuned LLM")
    ax.fill_between(x, llm_corr - llm_corr_sd, llm_corr + llm_corr_sd,
                    color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(x, gbm_corr, color=ORANGE, linewidth=2, marker="s", markersize=5,
            label="HistGB")
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("Pearson correlation")
    ax.set_title("(a) Individual-level signal")
    ax.set_ylim(-0.25, 0.9)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    ax.plot(x, llm_mae, color=BLUE, linewidth=2, marker="o", markersize=5,
            label="Fine-tuned LLM")
    ax.fill_between(x, llm_mae - llm_mae_sd, llm_mae + llm_mae_sd,
                    color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(x, gbm_mae, color=ORANGE, linewidth=2, marker="s", markersize=5,
            label="HistGB")
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("MAE")
    ax.set_title("(b) Absolute error")
    ax.set_ylim(14, 42)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")

    fig.supxlabel("ANES survey year", y=-0.02, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig2_temporal_decay")


def fig_fas() -> None:
    table = json.load(open(config.RESULTS / "paper_tables" / "gss_fas.json"))
    configs = ["EDC", "EHC"]
    runs = [("base", "Base", GRAY), ("sft", "SFT", BLUE),
            ("noneRAG+sft-v3", "SFT (best)", AQUA)]
    metrics = [("marginal_js", "Marginal JS"), ("joint_js", "Joint JS")]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    width = 0.24
    x = np.arange(len(configs))
    for ax, (metric, title) in zip(axes, metrics):
        for i, (key, label, color) in enumerate(runs):
            vals = [table[f"{c}|{key}"][metric] for c in configs]
            pos = x + (i - 1) * (width + 0.02)
            bars = ax.bar(pos, vals, width=width, color=color, label=label,
                          edgecolor="white", linewidth=1.0)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(configs)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.set_ylim(0, 0.135 if metric == "joint_js" else 0.07)
    axes[0].legend(frameon=False, loc="upper right", ncol=1)
    fig.tight_layout()
    _save(fig, "fig3_fas_js")


def fig_scatter() -> None:
    df = pd.read_json(config.PREDS / "qwen3_t_seed42_test_r_generate.predictions.jsonl",
                      lines=True)
    d = df[df["year"].isin([2016, 2018, 2019])]
    corr = np.corrcoef(d["true_score"], d["pred_float"])[0, 1]
    mae = np.mean(np.abs(d["true_score"] - d["pred_float"]))

    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    hb = ax.hexbin(d["true_score"], d["pred_float"], gridsize=35, mincnt=1,
                   cmap="Blues", linewidths=0.2)
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("Ground-truth thermometer score")
    ax.set_ylabel("Predicted score")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.text(0.04, 0.96, f"r = {corr:.3f}\nMAE = {mae:.2f}\nn = {len(d):,}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8, color=INK)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("count", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    _save(fig, "fig4_pas_scatter")


def main() -> None:
    fig_temporal()
    fig_fas()
    fig_scatter()


if __name__ == "__main__":
    main()
