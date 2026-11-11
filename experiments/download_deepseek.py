"""Download DeepSeek-R1-Distill-Llama-8B (ungated) for the second-family check."""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEST = Path("/home/jpzhao_team/models/deepseek-r1-distill-llama-8b")
REPO = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"repo={REPO}\ndest={DEST}", flush=True)
    path = snapshot_download(
        repo_id=REPO,
        local_dir=str(DEST),
        token=os.environ.get("HF_TOKEN"),
        resume_download=True,
    )
    print(f"downloaded to {path}", flush=True)


if __name__ == "__main__":
    main()
