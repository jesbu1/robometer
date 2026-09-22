#!/usr/bin/env python3
"""Upload one tar part to a HF dataset repo, with retries. Reads chunk on stdin.

Used by scripts/tar_and_upload_processed.sh via `split --filter`; split sets
$FILE to the intended part filename and pipes the chunk to this script.
"""
import os
import sys
import time

from huggingface_hub import HfApi

MAX_ATTEMPTS = 6


def main() -> int:
    file_path = os.environ["FILE"]
    repo_id = sys.argv[1]
    basename = os.path.basename(file_path)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    print(f"[uploader] receiving {basename} ...", flush=True)
    with open(file_path, "wb") as f:
        chunk_size = 8 * 1024 * 1024
        while True:
            buf = sys.stdin.buffer.read(chunk_size)
            if not buf:
                break
            f.write(buf)
    size_gb = os.path.getsize(file_path) / 1024**3
    print(f"[uploader] {basename} received ({size_gb:.1f} GiB), uploading...", flush=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=basename,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add {basename}",
            )
            os.remove(file_path)
            print(f"[uploader] DONE {basename}", flush=True)
            return 0
        except Exception as e:  # noqa: BLE001
            wait = min(600, 60 * attempt)
            print(f"[uploader] attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}; retry in {wait}s", flush=True)
            time.sleep(wait)
    print(f"[uploader] FAILED permanently: {basename}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
