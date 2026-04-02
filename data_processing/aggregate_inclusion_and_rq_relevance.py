#!/usr/bin/env python3
"""Aggregate GPT-5 judgements into inclusion scores.

This script reads batch CSVs that contain `gpt5...judgementN` columns
and an `excerpt_id` column. It aggregates across all `gpt5` judgement
columns to produce `quality_criterion` for each excerpt.

If a corresponding no-context CSV exists (same basename in the
`--noctx-dir`), the script will copy its judgement columns as
`nc_rqN` and also add `avg_nc` and `max_nc` which aggregate across the
no-context judgement columns.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict

import pandas as pd



def read_csv_with_fallback(path: str) -> pd.DataFrame:
    """Read a CSV trying several encodings until one works.

    Tries utf-8, utf-8-sig, latin-1, cp1252. Raises the last exception
    if none succeed.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # broad catch to try alternatives
            last_err = e
    # if we get here, re-raise a UnicodeDecodeError if present, else last
    raise last_err


def build_judgement_column_index(batch_df: pd.DataFrame) -> Dict[int, str]:
    pattern = re.compile(r'gpt5[_\s\-]*judgement[_\s\-]*(\d+)', re.I)
    mapping: Dict[int, str] = {}
    for col in batch_df.columns:
        m = pattern.search(col)
        if m:
            try:
                idx = int(m.group(1))
            except Exception:
                continue
            mapping[idx] = col
    return mapping


def canonical_aggregated_basename(input_basename: str) -> str:
    """Return canonical aggregated_inclusion basename for an input CSV filename."""
    root, ext = os.path.splitext(input_basename)
    if root.endswith('__reference-based_metrics__inclusion'):
        base = root[:-len('__reference-based_metrics__inclusion')]
    elif root.endswith('__reference-based_metrics__aggregated_inclusion'):
        base = root[:-len('__reference-based_metrics__aggregated_inclusion')]
    elif root.endswith('__inclusion'):
        base = root[:-len('__inclusion')]
    elif root.endswith('__aggregated_inclusion'):
        base = root[:-len('__aggregated_inclusion')]
    else:
        base = root
    return f'{base}__reference-based_metrics__aggregated_inclusion{ext}'


def aggregate_judgements_for_file(batch_csv: str, out_csv: str, noctx_csv: str | None = None) -> None:
    # read with encoding fallback to avoid UnicodeDecodeError on non-UTF8 files
    batch_df = read_csv_with_fallback(batch_csv)

    # find all judgement columns in the batch file
    judgement_columns_by_index = build_judgement_column_index(batch_df)
    judgement_columns = [
        judgement_columns_by_index[k]
        for k in sorted(judgement_columns_by_index.keys())
    ]

    # compute aggregated inclusion columns across all judgements
    excerpt_col = 'excerpt_id'
    if excerpt_col not in batch_df.columns:
        raise KeyError(f"Expected column 'excerpt_id' in {batch_csv}")
    out_df = pd.DataFrame()
    out_df[excerpt_col] = batch_df[excerpt_col]

    if judgement_columns:
        numeric_judgements = batch_df[judgement_columns].apply(pd.to_numeric, errors='coerce')
        out_df['quality_criterion'] = numeric_judgements.max(axis=1, skipna=True)
    else:
        out_df['quality_criterion'] = pd.NA

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # If a no-context CSV is provided, copy its judgement columns and add
    # aggregated `avg_nc` / `max_nc` columns.
    if noctx_csv and os.path.exists(noctx_csv):
        noctx_df = read_csv_with_fallback(noctx_csv)
        no_context_columns_by_index = build_judgement_column_index(noctx_df)
        no_context_columns = [
            no_context_columns_by_index[k]
            for k in sorted(no_context_columns_by_index.keys())
        ]
        for idx in sorted(no_context_columns_by_index.keys()):
            original_column = no_context_columns_by_index[idx]
            output_column = f'nc_rq{idx}'
            out_df[output_column] = noctx_df[original_column]

        if no_context_columns:
            no_context_numeric = noctx_df[no_context_columns].apply(pd.to_numeric, errors='coerce')
            out_df['avg_nc'] = no_context_numeric.mean(axis=1, skipna=True)
            out_df['max_nc'] = no_context_numeric.max(axis=1, skipna=True)
        else:
            out_df['avg_nc'] = pd.NA
            out_df['max_nc'] = pd.NA

    out_df.to_csv(out_csv, index=False)
    print(f'Wrote: {out_csv}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate gpt5 judgements into inclusion summaries')
    parser.add_argument('--batch-dir', default='data/reference-based_metrics/inclusion')
    parser.add_argument('--out-dir', default='data/reference-based_metrics/aggregated_inclusion')
    parser.add_argument('--suffix', default='.csv')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--noctx-dir', default='', help='Directory for no-context CSV files')
    args = parser.parse_args()

    batch_dir = args.batch_dir
    out_dir = args.out_dir

    batch_files = [f for f in os.listdir(batch_dir) if f.lower().endswith('.csv')]
    if not batch_files:
        print('No CSV files found in', batch_dir)
        return

    for fname in batch_files:
        batch_path = os.path.join(batch_dir, fname)
        out_basename = canonical_aggregated_basename(fname)
        # Keep --suffix behavior for backward compatibility while preserving
        # canonical basename semantics.
        out_base_without_ext = os.path.splitext(out_basename)[0]
        out_path = os.path.join(out_dir, out_base_without_ext + args.suffix)
        # look for a corresponding no-context file
        noctx_path = None
        if args.noctx_dir:
            candidate = os.path.join(args.noctx_dir, fname)
            if os.path.exists(candidate):
                noctx_path = candidate
        if args.dry_run:
            print(f'DRY RUN: would process {batch_path}'
                  f'{" + " + noctx_path if noctx_path else ""} -> {out_path}')
            continue
        aggregate_judgements_for_file(batch_path, out_path, noctx_path)


if __name__ == '__main__':
    main()
