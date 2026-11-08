"""Build body.md from the chapter markdown, then convert with pandoc.

- strips section numbers (LaTeX numbers sections itself)
- converts author-year citations to pandoc/natbib keys
- adds table captions and labels
- inserts figure includes with labels
"""
from __future__ import annotations

import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CH = BASE / "docs" / "04_论文稿件" / "chapters"
OUT = Path(__file__).resolve().parent / "body.md"

CITES = {
    "Argyle et al., 2022": "argyle2022",
    "Dillion et al., 2023": "dillion2023",
    "Kim and Lee, 2023": "kim2023",
    "Santurkar et al., 2023": "santurkar2023",
    "Sun et al., 2024": "sun2024",
    "Zhou et al., 2024": "zhou2024",
    "Zhao et al., 2025": "zhao2025",
    "Cao et al., 2025": "cao2025",
    "Park et al., 2024": "park2024",
    "Wang et al., 2024": "wang2024",
    "Xu et al., 2024": "xu2024",
    "Salemi et al., 2024": "salemi2024",
    "Ng et al., 2024": "ng2024",
    "Chu et al., 2023": "chu2023",
    "Zhao et al., 2023": "zhao2023",
    "Deming and Stephan, 1940": "deming1940",
    "Fienberg, 1970": "fienberg1970",
    "Cleveland, 1979": "cleveland1979",
    "Ma et al., 2024": "ma2024",
    "Mou et al., 2024": "mou2024",
    "Ziems et al., 2024": "ziems2024",
    "Anthis et al., 2025": "anthis2025",
}

TABLE1 = """
| Item | Value |
|---|---|
| Raw rows | 132,948 |
| Clean rows | 98,159 |
| Respondents | 3,165 |
| Respondents in more than one wave | 3,080 |
| Test-only respondents | 0 |
| `train_t` / `val_t` / `test_t` | 6,516 / 2,501 / 3,421 |
| `train_r` / `test_r` / `test_tr` | 5,094 / 6,014 / 226 |
: Data audit and split statistics.\\label{tab:audit}
"""

CAPTIONS = {
    "05_pas.md": [
        ": PAS in-distribution results on unseen respondents (2016--2019).\\label{tab:pas}",
        ": PAS by survey year (fine-tuned LLM vs. HistGB).\\label{tab:year}",
    ],
    "06_fas.md": [
        ": FAS distributional alignment on GSS.\\label{tab:fas}",
    ],
    "08_ablation.md": [
        ": Ablations and cross-pressure subgroups (2016--2019).\\label{tab:ablation}",
    ],
}

FIGURES = {
    "05_pas.md": (
        "motivates the anchor readout discussed next.",
        "\n![Predicted score versus ground-truth thermometer score on the in-distribution rows (seed 42). The dashed line is $y=x$; the density is compressed toward 50.](figures/fig4_pas_scatter.pdf){width=0.55\\linewidth}\n: In-distribution prediction density.\\label{fig:scatter}\n",
    ),
    "06_fas.md": (
        "we scope the \"80% reduction\" claim to EDC.",
        "\n![Marginal and joint JS divergence for EDC and EHC. Lower is better.](figures/fig3_fas_js.pdf){width=\\linewidth}\n: FAS distributional alignment.\\label{fig:fas}\n",
    ),
    "07_temporal.md": (
        "MAE rises from 18.85 to 23.41, 27.98, and 35.52.",
        "\n![Per-year correlation (a) and MAE (b) for the fine-tuned LLM and HistGB. Shaded bands are standard deviations over three seeds.](figures/fig2_temporal_decay.pdf){width=\\linewidth}\n: Temporal decay of individual-level signal.\\label{fig:temporal}\n",
    ),
}


def convert_citations(text: str) -> str:
    # Narrative citations first: "Argyle et al. (2022)" -> "@argyle2022"
    for plain, key in CITES.items():
        author, year = plain.rsplit(", ", 1)
        text = text.replace(f"{author} ({year})", f"@{key}")

    # Parenthetical groups: "(A et al., 2022; B, 2023)" -> "[@a; @b]"
    def repl(match: re.Match) -> str:
        inner = match.group(1)
        parts = [p.strip() for p in inner.split(";")]
        keys = []
        for p in parts:
            if p not in CITES:
                return match.group(0)
            keys.append("@" + CITES[p])
        return "[" + "; ".join(keys) + "]"

    return re.sub(r"\(([^()]*?(?:19|20)\d{2}[^()]*?)\)", repl, text)


def add_table_captions(text: str, filename: str) -> str:
    captions = CAPTIONS.get(filename, [])
    lines = text.splitlines()
    out, i, count = [], 0, 0
    while i < len(lines):
        if lines[i].startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            out.extend(block)
            if count < len(captions):
                out.append(captions[count])
            count += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def main() -> None:
    order = ["01_intro.md", "02_related.md", "03_data.md", "04_method.md",
             "05_pas.md", "06_fas.md", "07_temporal.md", "08_ablation.md",
             "09_limitations.md", "10_conclusion.md"]
    parts = []
    for name in order:
        text = (CH / name).read_text(encoding="utf-8")
        # strip "## 3 " / "### 3.2 " numbering
        text = re.sub(r"^##\s+\d+\s+", "## ", text, flags=re.M)
        text = re.sub(r"^###\s+\d+(?:\.\d+)*\s+", "### ", text, flags=re.M)
        text = convert_citations(text)

        if name == "03_data.md":
            text = text.replace(
                "Table 1 summarizes the audit.",
                "Table 1 summarizes the audit.\n" + TABLE1,
            )
        text = add_table_captions(text, name)

        anchor = FIGURES.get(name)
        if anchor:
            needle, figure = anchor
            if needle in text:
                text = text.replace(needle, needle + figure, 1)
            else:
                print(f"WARNING: figure anchor not found in {name}")

        parts.append(text)

    OUT.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT} ({len(OUT.read_text().split())} words)")


if __name__ == "__main__":
    main()
