"""Coverage-aware FAS summary.

The generated CSVs contain 351-400 of the 400 requested rows. JS is computed on
the returned rows only, so a run that returns fewer rows is not penalized. We
report coverage and a conservative adjusted JS that charges the missing mass as
maximally wrong (JS max = ln 2).

Outputs:
    experiments/results/paper_tables/fas_coverage.json
    experiments/results/paper_tables/fas_coverage.md
"""
from __future__ import annotations

import json
import math

from experiments import config

TARGET = 400.0
JS_MAX = math.log(2.0)


def main() -> None:
    table = json.load(open(config.RESULTS / "paper_tables" / "gss_fas.json"))
    out = {}
    for key, r in table.items():
        coverage = r["n_gen"] / TARGET
        missing = 1.0 - coverage
        out[key] = {
            **r,
            "coverage": coverage,
            "missing": missing,
            "adjusted_marginal_js": r["marginal_js"] + missing * JS_MAX,
            "adjusted_joint_js": r["joint_js"] + missing * JS_MAX,
        }

    out_dir = config.RESULTS / "paper_tables"
    with open(out_dir / "fas_coverage.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    lines = ["# FAS coverage-adjusted metrics", "",
             "| config | run | n | coverage | marginal JS | adj. marginal JS | joint JS | adj. joint JS |",
             "|---|---|---|---|---|---|---|---|"]
    for key, r in out.items():
        name, label = key.split("|")
        lines.append(
            f"| {name} | {label} | {r['n_gen']} | {r['coverage']:.3f} | "
            f"{r['marginal_js']:.4f} | {r['adjusted_marginal_js']:.4f} | "
            f"{r['joint_js']:.4f} | {r['adjusted_joint_js']:.4f} |"
        )
    (out_dir / "fas_coverage.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
