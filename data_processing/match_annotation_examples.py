#!/usr/bin/env python3
"""Extract matched LLM judgements for annotation comparison.

For each item in annotation_items.json:
1) Inclusion judgement:
   - Find task `{item_id}_inclusion`
   - Determine which results-section index applies by matching task text against
	 `data/results/{stem}.txt`, split by `%&%`
   - Use that index to select `gpt5_judgement_{idx}` from
	 `data/reference-based_metrics/inclusion/{stem}.csv` for the same
	 excerpt id.

2) RQ relevance judgement:
   - Find task `{item_id}_rq_relevance`
   - Determine which research-question index applies by matching task text against
	 `data/research_questions/{stem}.txt`, split by newline
   - Use that index to select `gpt5_judgement_{idx}` from
	 `data/reference-free_metrics/rq_relevance/{stem}.csv` for the same excerpt id.

Outputs one CSV with one row per annotation item.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


DEFAULT_ITEMS_JSON = Path("data/annotation_task/annotation_items.json")
DEFAULT_INCLUSION_DIR = Path("data/reference-based_metrics/inclusion")
DEFAULT_INCLUSION_TEXT_DIR = Path("data/results")
DEFAULT_RQ_DIR = Path("data/reference-free_metrics/rq_relevance")
DEFAULT_RQ_TEXT_DIR = Path("data/research_questions")
DEFAULT_OUTPUT = Path("data/annotation_task/llm_judgements_for_annotation_comparison.csv")


def canonical_stem(stem: str) -> str:
	return stem.split("__", 1)[0]


def resolve_companion_text(directory: Path, stem: str) -> Path:
	exact = directory / f"{stem}.txt"
	if exact.exists():
		return exact

	base = canonical_stem(stem)
	base_exact = directory / f"{base}.txt"
	if base_exact.exists():
		return base_exact

	candidates = sorted(p for p in directory.glob(f"{base}__*.txt") if p.is_file())
	if candidates:
		return sorted(candidates, key=lambda p: (len(p.name), p.name))[0]

	return exact


def normalize_text(text: str) -> str:
	normalized = unicodedata.normalize("NFKC", text)
	normalized = normalized.lower()
	normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
	normalized = normalized.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-").replace("—", "-")
	normalized = re.sub(r"[>#*_`]+", " ", normalized)
	normalized = re.sub(r"\s+", " ", normalized)
	return normalized.strip()


def safe_read_text(path: Path) -> str:
	return path.read_text(encoding="utf-8", errors="replace")


def load_items(path: Path) -> List[Dict[str, Any]]:
	data = json.loads(safe_read_text(path))
	if isinstance(data, list):
		return data
	if isinstance(data, dict):
		if isinstance(data.get("items"), list):
			return data["items"]
		for value in data.values():
			if isinstance(value, list):
				return value
	return []


def get_task(item: Dict[str, Any], suffix: str) -> Optional[Dict[str, Any]]:
	item_id = str(item.get("id", "")).strip()
	target = f"{item_id}_{suffix}"
	tasks = item.get("tasks", [])
	if isinstance(tasks, list):
		for task in tasks:
			if isinstance(task, dict) and str(task.get("id", "")).strip() == target:
				return task
	return None


def task_search_text(task: Optional[Dict[str, Any]]) -> str:
	if not task:
		return ""
	parts: List[str] = []
	for key in ("Content", "content", "supplementary_information", "supplementaryInformation"):
		value = task.get(key)
		if isinstance(value, str) and value.strip():
			parts.append(value)
	return "\n\n".join(parts)


def load_results_sections(path: Path) -> List[str]:
	text = safe_read_text(path)
	sections = [section.strip() for section in text.split("%&%") if section.strip()]
	return sections


def load_research_questions(path: Path) -> List[str]:
	text = safe_read_text(path)
	lines = [line.strip() for line in text.splitlines() if line.strip()]
	return lines


def choose_match_index(search_text: str, candidates: List[str]) -> Optional[int]:
	if not search_text or not candidates:
		return None

	haystack = normalize_text(search_text)
	normalized_candidates = [normalize_text(candidate) for candidate in candidates]

	# Strict containment first.
	for idx, candidate in enumerate(normalized_candidates, start=1):
		if candidate and candidate in haystack:
			return idx

	# Fallback: choose highest token-overlap candidate when partial inclusion exists.
	hay_tokens = set(re.findall(r"[a-z0-9']+", haystack))
	best_idx: Optional[int] = None
	best_score = 0.0
	for idx, candidate in enumerate(normalized_candidates, start=1):
		cand_tokens = set(re.findall(r"[a-z0-9']+", candidate))
		if not cand_tokens:
			continue
		overlap = len(hay_tokens & cand_tokens) / len(cand_tokens)
		if overlap > best_score:
			best_score = overlap
			best_idx = idx

	if best_idx is not None and best_score >= 0.6:
		return best_idx
	return None


def read_metric_csvs(directory: Path) -> Dict[str, pd.DataFrame]:
	by_stem: Dict[str, pd.DataFrame] = {}
	for csv_path in sorted(directory.glob("*.csv")):
		try:
			df = pd.read_csv(csv_path)
		except Exception as exc:
			print(f"Warning: could not read {csv_path}: {exc}")
			continue
		if "excerpt_id" not in df.columns:
			print(f"Warning: skipping {csv_path}, missing excerpt_id")
			continue
		df = df.copy()
		df["excerpt_id"] = pd.to_numeric(df["excerpt_id"], errors="coerce").astype("Int64")
		base = canonical_stem(csv_path.stem)
		by_stem[base] = df
		by_stem[csv_path.stem] = df
	return by_stem


def select_judgement(df: pd.DataFrame, excerpt_id: int, judgement_idx: Optional[int]) -> Optional[Any]:
	if judgement_idx is None:
		return None
	column = f"gpt5_judgement_{judgement_idx}"
	if column not in df.columns:
		return None
	row = df.loc[df["excerpt_id"] == excerpt_id]
	if row.empty:
		return None
	value = row.iloc[0][column]
	if pd.isna(value):
		return None
	return value


def process_items(
	items: List[Dict[str, Any]],
	inclusion_metric_by_stem: Dict[str, pd.DataFrame],
	rq_metric_by_stem: Dict[str, pd.DataFrame],
	inclusion_text_dir: Path,
	rq_text_dir: Path,
) -> pd.DataFrame:
	records: List[Dict[str, Any]] = []

	for item in items:
		item_id_raw = str(item.get("id", "")).strip()
		stem = canonical_stem(str(item.get("stem", "")).strip())

		excerpt_id: Optional[int]
		try:
			excerpt_id = int(item_id_raw)
		except ValueError:
			excerpt_id = None

		inclusion_task = get_task(item, "inclusion")
		rq_task = get_task(item, "rq_relevance")

		inclusion_search_text = task_search_text(inclusion_task)
		rq_search_text = task_search_text(rq_task)

		inclusion_idx: Optional[int] = None
		rq_idx: Optional[int] = None

		inclusion_text_path = resolve_companion_text(inclusion_text_dir, stem)
		if inclusion_text_path.exists() and inclusion_search_text:
			inclusion_sections = load_results_sections(inclusion_text_path)
			inclusion_idx = choose_match_index(inclusion_search_text, inclusion_sections)

		rq_text_path = resolve_companion_text(rq_text_dir, stem)
		if rq_text_path.exists() and rq_search_text:
			rq_lines = load_research_questions(rq_text_path)
			rq_idx = choose_match_index(rq_search_text, rq_lines)

		inclusion_judgement = None
		if stem in inclusion_metric_by_stem and excerpt_id is not None:
			inclusion_judgement = select_judgement(inclusion_metric_by_stem[stem], excerpt_id, inclusion_idx)

		rq_judgement = None
		if stem in rq_metric_by_stem and excerpt_id is not None:
			rq_judgement = select_judgement(rq_metric_by_stem[stem], excerpt_id, rq_idx)

		records.append(
			{
				"item_id": item_id_raw,
				"stem": stem,
				"inclusion_match_index": inclusion_idx,
				"inclusion_judgement": inclusion_judgement,
				"rq_match_index": rq_idx,
				"rq_relevance_judgement": rq_judgement,
			}
		)

	return pd.DataFrame(records)


def main(
	items_json: Path,
	inclusion_dir: Path,
	inclusion_text_dir: Path,
	rq_dir: Path,
	rq_text_dir: Path,
	output_csv: Path,
) -> None:
	if not items_json.exists():
		raise SystemExit(f"Input not found: {items_json}")

	items = load_items(items_json)
	if not items:
		raise SystemExit("No items found in annotation JSON.")

	inclusion_metric_by_stem = read_metric_csvs(inclusion_dir)
	rq_metric_by_stem = read_metric_csvs(rq_dir)

	out_df = process_items(
		items=items,
		inclusion_metric_by_stem=inclusion_metric_by_stem,
		rq_metric_by_stem=rq_metric_by_stem,
		inclusion_text_dir=inclusion_text_dir,
		rq_text_dir=rq_text_dir,
	)

	output_csv.parent.mkdir(parents=True, exist_ok=True)
	out_df.to_csv(output_csv, index=False)

	total = len(out_df)
	inclusion_ok = out_df["inclusion_judgement"].notna().sum()
	rq_ok = out_df["rq_relevance_judgement"].notna().sum()
	print(f"Wrote {output_csv}")
	print(f"Rows: {total}")
	print(f"Inclusion matches with judgement: {inclusion_ok}/{total}")
	print(f"RQ relevance matches with judgement: {rq_ok}/{total}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Extract matched inclusion and RQ-relevance judgements for annotation items")
	parser.add_argument("--items-json", default=str(DEFAULT_ITEMS_JSON), help="Path to annotation_items.json")
	parser.add_argument("--inclusion-dir", default=str(DEFAULT_INCLUSION_DIR), help="Directory with inclusion metric CSV files")
	parser.add_argument("--inclusion-text-dir", default=str(DEFAULT_INCLUSION_TEXT_DIR), help="Directory with paper results text files")
	parser.add_argument("--rq-dir", default=str(DEFAULT_RQ_DIR), help="Directory with rq_relevance metric CSV files")
	parser.add_argument("--rq-text-dir", default=str(DEFAULT_RQ_TEXT_DIR), help="Directory with research question text files")
	parser.add_argument("--out", "-o", default=str(DEFAULT_OUTPUT), help="Output CSV path")
	args = parser.parse_args()

	main(
		items_json=Path(args.items_json),
		inclusion_dir=Path(args.inclusion_dir),
		inclusion_text_dir=Path(args.inclusion_text_dir),
		rq_dir=Path(args.rq_dir),
		rq_text_dir=Path(args.rq_text_dir),
		output_csv=Path(args.out),
	)
