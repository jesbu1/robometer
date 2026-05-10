"""
Download the MolmoAct2 Bimanual YAM dataset collection from HuggingFace.
Uses HF_TOKEN env var for authentication.
"""

import os
from huggingface_hub import get_collection, snapshot_download
from tqdm import tqdm

token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable is required")

cache_dir = "/data/molmoact2_data"

collection = get_collection(
    "allenai/molmoact2-bimanualyam-dataset",
    token=token,
)

for item in tqdm(collection.items):
    if item.item_type != "dataset":
        continue
    dataset_id = item.item_id
    try:
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            cache_dir=cache_dir,
            token=token,
            resume_download=True,
        )
        print(f"Downloaded: {dataset_id}")
    except Exception as e:
        print(f"Failed: {dataset_id} -> {e}")
