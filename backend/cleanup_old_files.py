#!/usr/bin/env python3
"""Clean up uploaded videos, crops, evidence, and reports older than N days.

Usage:
    python cleanup_old_files.py              # dry run (default 2 days)
    python cleanup_old_files.py --apply      # actually delete
    python cleanup_old_files.py --days 7     # older than 7 days
"""

import argparse
import shutil
import time
from pathlib import Path

# Default to the production data dir, fallback to local uploads/
DATA_DIR = Path("/app/data/uploads") if Path("/app/data/uploads").exists() else Path("uploads")


def find_old_dirs(base: Path, max_age_days: int) -> list[tuple[Path, float]]:
    """Find subdirectories whose newest file is older than max_age_days."""
    cutoff = time.time() - (max_age_days * 86400)
    old = []
    if not base.exists():
        return old
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        # Find the newest file in this directory tree
        newest = 0.0
        for f in child.rglob("*"):
            if f.is_file():
                mtime = f.stat().st_mtime
                if mtime > newest:
                    newest = mtime
        if newest > 0 and newest < cutoff:
            old.append((child, newest))
    return old


def get_dir_size(path: Path) -> int:
    """Total bytes in a directory tree."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Clean up old scan files")
    parser.add_argument("--days", type=int, default=2, help="Delete files older than N days (default: 2)")
    parser.add_argument("--apply", action="store_true", help="Actually delete (default is dry run)")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] Cleaning files older than {args.days} days from {data_dir}")
    print()

    # Directories to scan:
    #   uploads/<user_id>/<scan_id>/  — videos
    #   uploads/crops/<scan_id>/      — crop images
    #   uploads/evidence/<scan_id>/   — evidence frames
    #   uploads/reports/<scan_id>/    — report JSON

    total_freed = 0

    # 1. User video directories (uploads/<user_id>/<scan_id>/)
    #    Two levels deep: user_id → scan_id
    for user_dir in sorted(data_dir.iterdir()) if data_dir.exists() else []:
        if not user_dir.is_dir() or user_dir.name in ("crops", "evidence", "reports"):
            continue
        for scan_dir, newest in find_old_dirs(user_dir, args.days):
            size = get_dir_size(scan_dir)
            age_days = (time.time() - newest) / 86400
            print(f"  VIDEO   {scan_dir.relative_to(data_dir)}  ({human_size(size)}, {age_days:.1f}d old)")
            total_freed += size
            if args.apply:
                shutil.rmtree(scan_dir)
        # Remove empty user dir
        if args.apply and user_dir.exists() and not any(user_dir.iterdir()):
            user_dir.rmdir()

    # 2-4. Flat scan_id dirs under crops/, evidence/, reports/
    for subdir_name in ("crops", "evidence", "reports"):
        subdir = data_dir / subdir_name
        for scan_dir, newest in find_old_dirs(subdir, args.days):
            size = get_dir_size(scan_dir)
            age_days = (time.time() - newest) / 86400
            label = subdir_name.upper().ljust(9)
            print(f"  {label} {scan_dir.relative_to(data_dir)}  ({human_size(size)}, {age_days:.1f}d old)")
            total_freed += size
            if args.apply:
                shutil.rmtree(scan_dir)

    print()
    verb = "Freed" if args.apply else "Would free"
    print(f"{verb}: {human_size(total_freed)}")
    if not args.apply and total_freed > 0:
        print("Run with --apply to delete.")


if __name__ == "__main__":
    main()
