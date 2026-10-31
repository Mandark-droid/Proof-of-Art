"""Publish only refreshed README cards to the four existing Pramaan dataset repositories."""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download

from scripts.push_hub_datasets import dataset_card, openenv_dataset_card

REPOSITORIES = (
    ("kshitijthakkar/pramaan-proof-of-art-eval-v2.1", lambda: dataset_card("eval", suite_name="eval-v2.1")),
    ("kshitijthakkar/pramaan-proof-of-art-eval-v2", lambda: dataset_card("eval", suite_name="eval-v2")),
    ("kshitijthakkar/pramaan-proof-of-art-train-v1", lambda: dataset_card("train", suite_name="train-v1")),
    ("kshitijthakkar/pramaan-proof-of-art-openenv", openenv_dataset_card),
)


def main() -> int:
    token = get_token()
    if not token:
        raise SystemExit("Log in to Hugging Face before updating dataset cards")
    api = HfApi(token=token)
    for repo_id, render in REPOSITORIES:
        card = render()
        current = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="README.md", revision="main")
        if Path(current).read_text(encoding="utf-8") == card:
            print(f"PASS {repo_id}: card already current")
            continue
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            revision="main",
            commit_message="Document Pramaan dataset schema and usage",
            operations=[CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=io.BytesIO(card.encode("utf-8")),
            )],
        )
        downloaded = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="README.md", revision="main")
        if Path(downloaded).read_text(encoding="utf-8") != card:
            raise RuntimeError(f"published README verification failed for {repo_id}")
        print(f"PASS {repo_id}: one normal README commit verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())