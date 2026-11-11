"""Download DeepSeek-R1-Distill-Llama-8B from ModelScope (multi-threaded)."""
from __future__ import annotations

from modelscope import snapshot_download

DEST = "/home/jpzhao_team/models/deepseek-r1-distill-llama-8b"
REPO = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def main() -> None:
    print(f"repo={REPO}\ndest={DEST}", flush=True)
    path = snapshot_download(
        REPO,
        revision="master",
        local_dir=DEST,
        max_workers=8,
    )
    print(f"downloaded to {path}", flush=True)


if __name__ == "__main__":
    main()
