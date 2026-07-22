import os
import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download RADAR auxiliary data from Hugging Face."
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="../",
        help="Local directory to save the downloaded auxiliary data."
    )
    parser.add_argument(
        "--include-patterns",
        type=str,
        nargs="*",
        default=None,
        help=(
            "All files in the repository will be downloaded."
        )
    )
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    snapshot_download(
        repo_id="radar-generalist/RADAR-auxiliary-data",
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=args.include_patterns,
        local_dir_use_symlinks=False
    )

    print(f"Auxiliary data download completed. Files are saved to: {args.local_dir}")


if __name__ == "__main__":
    main()


