"""Download Llama-3.1-8B-Instruct with the Python API (the CLI shebang is stale)."""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEST = Path("/home/jpzhao_team/models/llama-3.1-8b-instruct")
REPO = "meta-llama/Llama-3.1-8B-Instruct"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    print(f"repo={REPO}\ndest={DEST}\ntoken={'set' if token else 'not set'}", flush=True)
    path = snapshot_download(
        repo_id=REPO,
        local_dir=str(DEST),
        token=token,
        resume_download=True,
    )
    print(f"downloaded to {path}", flush=True)


if __name__ == "__main__":
    main()
