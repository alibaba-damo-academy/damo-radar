import os
import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download RADAR model checkpoints and support files from Hugging Face."
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="../ckpt",
        help="Local directory to save the downloaded checkpoints."
    )
    parser.add_argument(
        "--include-patterns",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional file patterns to download, e.g. "
            "'*.pth' or 'checkpoint_radar_plus.pth'. "
            "If not specified, all files in the repository will be downloaded."
        )
    )
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    snapshot_download(
        repo_id="radar-generalist/RADAR",
        repo_type="model",
        local_dir=args.local_dir,
        allow_patterns=args.include_patterns,
        local_dir_use_symlinks=False
    )

    print(f"Checkpoint download completed. Files are saved to: {args.local_dir}")


if __name__ == "__main__":
    main()

