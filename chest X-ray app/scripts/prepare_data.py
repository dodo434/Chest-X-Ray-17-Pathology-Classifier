"""Extract the local dataset archive once and create a reusable data directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_manager import extract_verified_archive, load_folder_dataset, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("archive.zip"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    extracted = extract_verified_archive(args.archive, args.output, overwrite=True)
    _, labels = load_folder_dataset(extracted)
    (args.output / "labels.json").write_text(
        json.dumps({"labels": labels, "archive_sha256": sha256_file(args.archive)}, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared {len(labels)} labels in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
