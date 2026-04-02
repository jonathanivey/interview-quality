#!/usr/bin/env python3
"""Compute token counts per excerpt aggregated by speaker.

For each mapping CSV in data/excerpt_transcripts/utterance_excerpt_pairs,
this script locates utterances in data/combined_transcripts, counts tokens
for each utterance, and writes an output CSV to
data/reference-free_metrics/token_counts with columns:

- excerpt_id
- participant_tokens
- interviewer_tokens

Tokenization uses `tiktoken` if available, otherwise falls back to
whitespace-splitting.
"""
from __future__ import annotations

import os
import sys
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import tiktoken


def get_tokenizer(name: str | None = None):
    try:
        if name:
            enc = tiktoken.get_encoding(name)
        else:
            # prefer cl100k_base where available
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = tiktoken.get_encoding("gpt2")
        return lambda s: len(enc.encode(s or ""))
    except Exception as e:
        print(f"Failed to initialize tiktoken encoder: {e}", file=sys.stderr)
        raise


def find_csv_files(root: str) -> list[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                out.append(os.path.join(dirpath, fn))
    return out


def build_utterance_index(combined_files: list[str], id_col: str = "utterance_id") -> dict:
    """Return mapping utterance_id -> dict with keys speaker, utterance, src_file."""
    idx: dict = {}
    for fp in combined_files:
        try:
            df = pd.read_csv(fp, dtype=str)
        except Exception:
            continue
        if id_col not in df.columns:
            continue
        # ensure columns we need exist
        for _, row in df.iterrows():
            uid = row.get(id_col)
            if pd.isna(uid) or uid is None:
                continue
            sid = str(uid)
            # coerce speaker and utterance to safe strings (replace NA with empty string)
            sp = row.get("speaker", "")
            if pd.isna(sp) or sp is None:
                sp = ""
            ut = row.get("utterance", "")
            if pd.isna(ut) or ut is None:
                ut = ""
            idx[sid] = {
                "speaker": sp,
                "utterance": ut,
                "src_file": fp,
            }
    return idx


def process_mapping(mapping_fp: str, utter_idx: dict, tokenizer, out_dir: str):
    try:
        map_df = pd.read_csv(mapping_fp, dtype=str)
    except Exception as e:
        print(f"Skipping mapping {mapping_fp}: cannot read CSV ({e})", file=sys.stderr)
        return

    if "utterance_id" not in map_df.columns or "excerpt_id" not in map_df.columns:
        print(f"Skipping mapping {mapping_fp}: missing required columns", file=sys.stderr)
        return

    # attach utterance text and speaker
    tokens_per_mapping = []
    missing = 0
    bad_rows = []
    base = os.path.splitext(os.path.basename(mapping_fp))[0]
    for _, row in map_df.iterrows():
        uid = str(row["utterance_id"])
        eid = row["excerpt_id"]
        info = utter_idx.get(uid)
        if not info:
            missing += 1
            continue
        text = info.get("utterance", "")
        speaker = (info.get("speaker") or "").strip().lower()
        # ensure we pass a string to tokenizer; catch and record any errors
        try:
            safe_text = text if isinstance(text, str) else str(text)
            count = tokenizer(safe_text)
        except Exception as e:
            bad_rows.append({"utterance_id": uid, "excerpt_id": eid, "speaker": speaker, "text_repr": repr(text), "error": str(e)})
            print(f"Error tokenizing utterance_id {uid} in {mapping_fp}: {e}", file=sys.stderr)
            count = 0
        tokens_per_mapping.append((eid, speaker, count))

    if missing:
        print(f"Warning: {missing} utterance_ids in {mapping_fp} not found in combined transcripts", file=sys.stderr)

    # aggregate by excerpt and speaker
    agg: dict[str, dict[str, int]] = defaultdict(lambda: {"participant": 0, "interviewer": 0})
    for eid, speaker, count in tokens_per_mapping:
        if speaker.startswith("participant") or speaker == "p" or speaker == "participant":
            agg[eid]["participant"] += int(count)
        else:
            # treat anything else as interviewer (more conservative)
            agg[eid]["interviewer"] += int(count)

    # create DataFrame
    rows = []
    for eid, vals in agg.items():
        rows.append({"excerpt_id": eid, "participant_tokens": vals["participant"], "interviewer_tokens": vals["interviewer"]})

    out_df = pd.DataFrame(rows)
    # ensure deterministic order
    out_df = out_df.sort_values("excerpt_id")

    # compute token ratio (participant / interviewer). If interviewer_tokens == 0, result will be NaN
    out_df["token_ratio"] = (
        out_df["participant_tokens"].astype(float) / out_df["interviewer_tokens"].replace(0, np.nan).astype(float)
    )

    base = os.path.splitext(os.path.basename(mapping_fp))[0]
    out_fp = os.path.join(out_dir, f"{base}.csv")
    out_df.to_csv(out_fp, index=False)
    print(f"Wrote {out_fp} ({len(out_df)} excerpts)")

    if bad_rows:
        bad_df = pd.DataFrame(bad_rows)
        out_bad_fp = os.path.join(out_dir, f"{base}_bad_utterances.csv")
        bad_df.to_csv(out_bad_fp, index=False)
        print(f"Wrote {out_bad_fp} ({len(bad_df)} problematic utterances).", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Compute token counts per excerpt aggregated by speaker")
    p.add_argument("--combined_dir", default="data/combined_transcripts", help="Folder with combined transcript CSVs")
    p.add_argument("--mappings_dir", default="data/excerpt_transcripts/utterance_excerpt_pairs", help="Folder with utterance->excerpt mapping CSVs")
    p.add_argument("--out_dir", default="data/reference-free_metrics/token_counts", help="Output folder for aggregated token counts")
    p.add_argument("--tiktoken_encoding", default=None, help="tiktoken encoding name (optional) e.g. cl100k_base or gpt2")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    combined_files = find_csv_files(args.combined_dir)
    if not combined_files:
        print(f"No combined transcripts found in {args.combined_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Indexing {len(combined_files)} combined transcript files...")
    utter_idx = build_utterance_index(combined_files)
    print(f"Indexed {len(utter_idx)} utterances")

    mapping_files = find_csv_files(args.mappings_dir)
    if not mapping_files:
        print(f"No mapping files found in {args.mappings_dir}", file=sys.stderr)
        sys.exit(1)

    tokenizer = get_tokenizer(args.tiktoken_encoding)

    for mf in mapping_files:
        process_mapping(mf, utter_idx, tokenizer, args.out_dir)


if __name__ == "__main__":
    main()
