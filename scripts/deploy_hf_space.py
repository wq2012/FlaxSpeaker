#!/usr/bin/env python3
"""Deploy `hf_space/` to a Hugging Face Space (Gradio SDK).

    python scripts/deploy_hf_space.py --repo_id wq2012/FlaxSpeaker

Uploads `app.py`, `README.md` (Space metadata front-matter), `requirements.txt` and the
`examples/` audio clips. The Space installs `flaxspeaker>=0.5.0` from PyPI (see
`hf_space/requirements.txt`) and downloads checkpoints from the Hub on first use, so
no weights are uploaded here.
"""

import argparse
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE_DIR = os.path.join(REPO_ROOT, "hf_space")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="wq2012/FlaxSpeaker")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi  # pytype: disable=import-error

    api = HfApi()
    space_exists = True
    try:
        api.repo_info(repo_id=args.repo_id, repo_type="space")
    except Exception:  # noqa: BLE001
        space_exists = False
    is_private = True if not space_exists else args.private
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=is_private,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=SPACE_DIR,
        ignore_patterns=["__pycache__/*", "*.pyc"],
        commit_message=(
            "Deploy FlaxSpeaker v0.5.0 Gradio demo "
            "(real inference, 49 models, per-model calibrated thresholds)"
        ),
    )
    print(f"Uploaded {SPACE_DIR} to https://huggingface.co/spaces/{args.repo_id}")
    if not space_exists:
        print(
            f"Newly created PRIVATE HF Space (switch to public when ready): "
            f"https://huggingface.co/spaces/{args.repo_id}"
        )


if __name__ == "__main__":
    main()
