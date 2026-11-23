"""Swap the architecture figure inside an existing deck, keeping all edits.

Usage:
    python docs/04_论文稿件/sync_deck_figure.py

Replaces any embedded image whose bytes match a previous fig1_pipeline render
with the current thesis/figures/fig1_pipeline.png, so hand edits to the deck
are preserved.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DECK = HERE / "导师阶段汇报_UMAC_2026-11-23.pptx"
FIG = ROOT / "thesis" / "figures" / "fig1_pipeline.png"

# hashes of earlier exports of the same figure (kept so old decks can be fixed)
STALE = {"7422868899c2", "12047aacd170", "2d433c23f25e", "82334ef3f50d"}


def main() -> None:
    fig = FIG.read_bytes()
    new_hash = hashlib.sha256(fig).hexdigest()
    tmp = DECK.with_suffix(".tmp.pptx")
    replaced = []
    with zipfile.ZipFile(DECK) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("ppt/media/"):
                h = hashlib.sha256(data).hexdigest()
                if h[:12] in STALE and h[:12] != new_hash[:12]:
                    data = fig
                    replaced.append(item.filename)
            zout.writestr(item, data)
    tmp.replace(DECK)
    print("replaced:", replaced or "nothing (already current)")


if __name__ == "__main__":
    main()
