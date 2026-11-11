"""Multi-threaded range downloader for the two DeepSeek safetensors shards.

ModelScope's single-stream download is ~0.5 MB/s; 16 range connections give
roughly an order of magnitude more throughput. Small files are already present
from the earlier snapshot downloads.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DEST = "/home/jpzhao_team/models/deepseek-r1-distill-llama-8b"
BASE = ("https://modelscope.cn/models/deepseek-ai/"
        "DeepSeek-R1-Distill-Llama-8B/resolve/master/")
FILES = [
    "model-00001-of-000002.safetensors",
    "model-00002-of-000002.safetensors",
]
CHUNK = 64 * 1024 * 1024
WORKERS = 16

_lock = threading.Lock()
_done = 0


def _size(url: str) -> int:
    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Range"].split("/")[1])


def _fetch(url: str, path: str, start: int, end: int) -> int:
    global _done
    last = None
    for attempt in range(3):
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                              stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(path, "r+b") as f:
                    f.seek(start)
                    n = 0
                    for chunk in r.iter_content(8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            n += len(chunk)
            with _lock:
                _done += n
            return n
        except Exception as exc:  # retry
            last = exc
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"{path} [{start}-{end}] failed: {last}")


def download(name: str) -> None:
    url = BASE + name
    path = os.path.join(DEST, name)
    size = _size(url)
    if os.path.exists(path) and os.path.getsize(path) == size:
        print(f"[skip] {name} already complete ({size})", flush=True)
        return
    with open(path, "wb") as f:
        f.truncate(size)
    ranges = [(i, min(i + CHUNK - 1, size - 1)) for i in range(0, size, CHUNK)]
    print(f"[start] {name}: {size/1e9:.2f} GB, {len(ranges)} chunks", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_fetch, url, path, s, e) for s, e in ranges]
        for fut in as_completed(futures):
            fut.result()
    print(f"[done] {name}: {os.path.getsize(path)} bytes", flush=True)


def main() -> None:
    os.makedirs(DEST, exist_ok=True)
    t0 = time.time()
    for name in FILES:
        download(name)
        print(f"elapsed {time.time()-t0:.1f}s, {_done/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
