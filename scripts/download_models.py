#!/usr/bin/env python3
"""Manage built-in models between the local models/ folder and HuggingFace (bangpc/infer-pack)."""

import argparse
import sys
from pathlib import Path

REPO_ID = "bangpc/infer-pack"
REPO_TYPE = "model"

# Files managed by this script, relative to PROJECT_ROOT / HuggingFace repo root
MODEL_FILES = [
    "models/face_detection_model/config.pbtxt",
    "models/face_detection_model/1/model.onnx",
    "models/face_feature_extraction_model/config.pbtxt",
    "models/face_feature_extraction_model/1/model.onnx",
]

# Root of the local project (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _require_hf_hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("       Run: pip install huggingface-hub")
        sys.exit(1)


def download_models(force: bool = False) -> None:
    hf = _require_hf_hub()

    for repo_path in MODEL_FILES:
        local_dest = PROJECT_ROOT / repo_path
        if local_dest.exists() and not force:
            print(f"  [skip]  {repo_path}  (already exists, use --force to re-download)")
            continue

        local_dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [fetch] {repo_path}")
        downloaded = hf.hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=repo_path,
            local_dir=str(PROJECT_ROOT),
        )
        print(f"          -> {Path(downloaded).relative_to(PROJECT_ROOT)}")

    print("\nAll models are ready.")


def upload_models(commit_message: str = "Update built-in models") -> None:
    hf = _require_hf_hub()

    # Check authentication before doing any work
    api = hf.HfApi()
    try:
        whoami = api.whoami()
        print(f"Authenticated as: {whoami['name']}")
    except hf.errors.LocalTokenNotFoundError:
        print("ERROR: not logged in to HuggingFace.")
        print("       Run: huggingface-cli login")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: authentication check failed: {exc}")
        print("       Run: huggingface-cli login")
        sys.exit(1)

    missing = [f for f in MODEL_FILES if not (PROJECT_ROOT / f).exists()]
    if missing:
        print("ERROR: the following files are missing locally and cannot be uploaded:")
        for f in missing:
            print(f"         {f}")
        sys.exit(1)

    print(f"Uploading models to {REPO_ID} ...")
    for repo_path in MODEL_FILES:
        local_src = PROJECT_ROOT / repo_path
        size_mb = local_src.stat().st_size / 1_048_576
        print(f"  [push]  {repo_path}  ({size_mb:.1f} MB)")
        try:
            api.upload_file(
                path_or_fileobj=str(local_src),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=commit_message,
            )
        except hf.errors.RepositoryNotFoundError:
            print(f"ERROR: repository '{REPO_ID}' not found or you lack write access.")
            print("       Make sure the repo exists and your token has write permission.")
            sys.exit(1)

    print(f"\nAll models uploaded to https://huggingface.co/{REPO_ID}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download or upload InferPack built-in models to/from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/download_models.py download
  python scripts/download_models.py download --force
  python scripts/download_models.py upload
  python scripts/download_models.py upload --message "Add optimized ArcFace weights"
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Pull models from HuggingFace into models/")
    dl.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local files already exist.",
    )

    up = sub.add_parser("upload", help="Push models/ to HuggingFace")
    up.add_argument(
        "--message",
        default="Update built-in models",
        metavar="MSG",
        help="Commit message for the HuggingFace upload (default: 'Update built-in models').",
    )

    args = parser.parse_args()

    if args.command == "download":
        print(f"Downloading models from {REPO_ID} ...")
        download_models(force=args.force)
    elif args.command == "upload":
        upload_models(commit_message=args.message)


if __name__ == "__main__":
    main()
