#!/usr/bin/env python3
"""
Helper script to rename files with duplicate basenames in data/ to have unique names, so that they are compatible with the QDR data deposit format.

Example:
    python data_processing/migrate_duplicate_filenames.py --dry-run
    python data_processing/migrate_duplicate_filenames.py --apply --rewrite-code
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TEXT_FILE_EXTS = {
    ".py",
    ".r",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".tsv",
    ".csv",
    ".ipynb",
}

# Default data subfolders where duplicated DOI-style basenames are typically
# used as linked companion files across pipeline stages.
DEFAULT_TARGET_DIRS = [
    "combined_transcripts",
    "context_blurbs",
    "excerpt_transcripts",
    "interviewer_techniques",
    "reference-based_metrics",
    "reference-free_metrics",
    "research_questions",
    "results",
]


@dataclass(frozen=True)
class RenamePlan:
    old_abs: Path
    new_abs: Path
    old_rel_from_repo: str
    new_rel_from_repo: str


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _should_skip(path: Path, data_root: Path, include_hidden: bool) -> bool:
    rel_parts = path.relative_to(data_root).parts
    if not include_hidden and any(part.startswith(".") for part in rel_parts):
        return True
    return False


def _make_parent_tag(data_root: Path, file_path: Path) -> str:
    parent_rel = file_path.parent.relative_to(data_root)
    parts = [part.replace(" ", "-") for part in parent_rel.parts]
    return "__".join(parts) if parts else "root"


def _build_plans(
    repo_root: Path,
    data_root: Path,
    target_dirs: List[str],
    include_hidden: bool,
) -> List[RenamePlan]:
    by_name: Dict[str, List[Path]] = {}
    target_roots = [(data_root / d) for d in target_dirs]
    for p in _iter_files(data_root):
        if _should_skip(p, data_root, include_hidden=include_hidden):
            continue
        if not any(str(p).startswith(str(root) + "/") or p == root for root in target_roots):
            continue
        by_name.setdefault(p.name, []).append(p)

    plans: List[RenamePlan] = []

    for _, files in sorted(by_name.items()):
        if len(files) <= 1:
            continue

        for old_abs in sorted(files):
            parent_tag = _make_parent_tag(data_root, old_abs)
            stem = old_abs.stem
            suffix = old_abs.suffix
            candidate_name = f"{stem}__{parent_tag}{suffix}"
            new_abs = old_abs.with_name(candidate_name)

            i = 2
            while new_abs.exists() and new_abs != old_abs:
                candidate_name = f"{stem}__{parent_tag}__{i}{suffix}"
                new_abs = old_abs.with_name(candidate_name)
                i += 1

            old_rel = old_abs.relative_to(repo_root).as_posix()
            new_rel = new_abs.relative_to(repo_root).as_posix()
            plans.append(
                RenamePlan(
                    old_abs=old_abs,
                    new_abs=new_abs,
                    old_rel_from_repo=old_rel,
                    new_rel_from_repo=new_rel,
                )
            )

    return plans


def _write_manifest_csv(manifest_path: Path, plans: List[RenamePlan]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["old_path", "new_path"])
        for plan in plans:
            writer.writerow([plan.old_rel_from_repo, plan.new_rel_from_repo])


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_FILE_EXTS


def _rewrite_code_literals(repo_root: Path, plans: List[RenamePlan], dry_run: bool) -> Tuple[int, int]:
    replacement_map = {
        plan.old_rel_from_repo: plan.new_rel_from_repo
        for plan in plans
    }

    files_changed = 0
    replacements_made = 0

    for p in _iter_files(repo_root):
        if p.is_relative_to(repo_root / "data"):
            continue
        if p.is_relative_to(repo_root / ".git"):
            continue
        if not _is_text_file(p):
            continue

        try:
            original = p.read_text(encoding="utf-8")
        except Exception:
            continue

        updated = original
        local_count = 0
        for old_rel, new_rel in replacement_map.items():
            if old_rel in updated:
                c = updated.count(old_rel)
                updated = updated.replace(old_rel, new_rel)
                local_count += c

        if local_count > 0:
            files_changed += 1
            replacements_made += local_count
            if not dry_run:
                p.write_text(updated, encoding="utf-8")

    return files_changed, replacements_made


def _apply_renames(plans: List[RenamePlan], dry_run: bool) -> int:
    count = 0
    for plan in plans:
        if plan.old_abs == plan.new_abs:
            continue
        count += 1
        if not dry_run:
            plan.old_abs.rename(plan.new_abs)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename duplicate data filenames and update code path references.")
    parser.add_argument("--repo-root", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--data-root", default="data", help="Data directory to scan (default: data)")
    parser.add_argument("--manifest", default="data/analysis/duplicate_filename_migration_map.csv", help="CSV manifest path")
    parser.add_argument(
        "--target-dirs",
        nargs="+",
        default=DEFAULT_TARGET_DIRS,
        help=(
            "Top-level subfolders under data/ to include in duplicate rename "
            "(default: combined_transcripts context_blurbs excerpt_transcripts "
            "interviewer_techniques reference-based_metrics "
            "reference-free_metrics research_questions results)"
        ),
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include dotfiles/directories (disabled by default)",
    )
    parser.add_argument("--apply", action="store_true", help="Apply file renames (default is dry-run)")
    parser.add_argument("--rewrite-code", action="store_true", help="Rewrite exact old path literals in repository text files")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing (default behavior if --apply is omitted)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = (repo_root / args.data_root).resolve()
    manifest_path = (repo_root / args.manifest).resolve()

    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    # Default behavior is dry-run unless --apply is provided.
    dry_run = (not args.apply) or args.dry_run

    plans = _build_plans(
        repo_root=repo_root,
        data_root=data_root,
        target_dirs=args.target_dirs,
        include_hidden=args.include_hidden,
    )

    print(f"Found {len(plans)} files to rename (duplicate basenames only).")
    for plan in plans[:30]:
        print(f"  {plan.old_rel_from_repo} -> {plan.new_rel_from_repo}")
    if len(plans) > 30:
        print(f"  ... and {len(plans) - 30} more")

    if not dry_run:
        _write_manifest_csv(manifest_path, plans)
    else:
        print(f"Dry-run: manifest would be written to {manifest_path.relative_to(repo_root).as_posix()}")

    renamed = _apply_renames(plans, dry_run=dry_run)
    if dry_run:
        print(f"Dry-run: would rename {renamed} files")
    else:
        print(f"Renamed {renamed} files")

    if args.rewrite_code:
        files_changed, replacements_made = _rewrite_code_literals(repo_root=repo_root, plans=plans, dry_run=dry_run)
        if dry_run:
            print(f"Dry-run: would update {replacements_made} path literals across {files_changed} files")
        else:
            print(f"Updated {replacements_made} path literals across {files_changed} files")


if __name__ == "__main__":
    main()
