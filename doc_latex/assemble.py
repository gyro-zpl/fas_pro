"""Insert figures/tables and resolve references in body.tex."""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BODY = HERE / "body.tex"
OUT = HERE / "body_final.tex"

WIDTHS = {
    "fig4_pas_scatter.pdf": "0.9\\columnwidth",
    "fig3_fas_js.pdf": "\\linewidth",
    "fig2_temporal_decay.pdf": "\\linewidth",
}
WIDE_FIGS = {"fig3_fas_js.pdf", "fig2_temporal_decay.pdf"}


def replace_figures(text: str) -> str:
    pattern = re.compile(
        r"\\includegraphics\{figures/([^}]+)\}\n: ([^\n]+?)\.\\label\{([^}]+)\}"
    )

    def repl(m: re.Match) -> str:
        fname, caption, label = m.group(1), m.group(2), m.group(3)
        width = WIDTHS.get(fname, "\\linewidth")
        env = "figure*" if fname in WIDE_FIGS else "figure"
        return (
            f"\\begin{{{env}}}[t]\n"
            "\\centering\n"
            f"\\includegraphics[width={width}]{{figures/{fname}}}\n"
            f"\\caption{{{caption}.}}\n"
            f"\\label{{{label}}}\n"
            f"\\end{{{env}}}"
        )

    return pattern.sub(repl, text)


def replace_refs(text: str) -> str:
    pairs = [
        ("Table 1", "Table~\\ref{tab:audit}"),
        ("Table 2", "Table~\\ref{tab:pas}"),
        ("Table 3", "Table~\\ref{tab:year}"),
        ("Table 4", "Table~\\ref{tab:fas}"),
        ("Table 5", "Table~\\ref{tab:ablation}"),
        ("Figure 2", "Figure~\\ref{fig:temporal}"),
        ("Figure 3", "Figure~\\ref{fig:fas}"),
        ("Figure 4", "Figure~\\ref{fig:scatter}"),
    ]
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def insert_tables(text: str) -> str:
    anchors = [
        ("Ridge is weaker at 22.47, 0.655, and 0.484.",
         "tables/table2_pas"),
        ("The LLM narrows the gap in 2019 but does not overtake the baseline.",
         "tables/table3_year"),
        ("claim to EDC.", "tables/table4_fas"),
        ("A3 reaches 21.45, 0.657, and 0.565.", "tables/table5_ablation"),
    ]
    for anchor, table in anchors:
        if anchor not in text:
            print(f"WARNING: table anchor not found: {anchor[:50]}")
            continue
        text = text.replace(anchor, anchor + f"\n\n\\input{{{table}}}\n", 1)
    return text


def fix_equations(text: str) -> str:
    sim_old = r"""\[
\text{sim}(v,c)=0.25\,\mathbb{1}[\text{party match}]
+0.25\,\mathbb{1}[\text{same ideological side}]
+0.20\,\mathbb{1}[\text{race match}]
+0.15\,\mathbb{1}[\text{gender match}]
+0.15\left(1-\frac{\lvert \text{age}_v-\text{age}_c\rvert}{100}\right).
\]"""
    sim_new = r"""\[
\begin{aligned}
\text{sim}(v,c) ={}& 0.25\,\mathbb{1}[\text{party match}] \\
&+0.25\,\mathbb{1}[\text{same ideological side}] \\
&+0.20\,\mathbb{1}[\text{race match}] \\
&+0.15\,\mathbb{1}[\text{gender match}] \\
&+0.15\left(1-\frac{\lvert \text{age}_v-\text{age}_c\rvert}{100}\right).
\end{aligned}
\]"""
    text = text.replace(sim_old, sim_new)

    # A few displays are a few points wider than a column; scale them.
    for start, end in [
        (r"\[" + "\n" + r"\mathcal{M}[c] =", r"\text{incumbent})," + "\n" + r"\]"),
        (r"\[" + "\n" + r"\text{party\_match}(v,c)=", r"\end{cases}" + "\n" + r"\]"),
        (r"\[" + "\n" + r"\text{ideo\_dist}(v,c)=", r"[0,6]," + "\n" + r"\]"),
    ]:
        i = text.find(start)
        if i == -1:
            continue
        j = text.find(end, i)
        if j == -1:
            continue
        block = text[i:j + len(end)]
        inner = block[len(r"\["):-len(r"\]")]
        replacement = (
            "\\begin{equation*}\n"
            "\\resizebox{\\columnwidth}{!}{$\\displaystyle " + inner + "$}\n"
            "\\end{equation*}"
        )
        text = text[:i] + replacement + text[j + len(end):]
    return text


def main() -> None:
    text = BODY.read_text(encoding="utf-8")
    # Pandoc emits a longtable for the audit table; ACL is two-column, so
    # replace it with a regular table generated from tables/table1_data.tex.
    text = re.sub(
        r"\\begin\{longtable\}.*?\\end\{longtable\}",
        "\\\\input{tables/table1_data}",
        text,
        flags=re.S,
    )
    text = replace_figures(text)
    text = replace_refs(text)
    text = insert_tables(text)
    text = fix_equations(text)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
