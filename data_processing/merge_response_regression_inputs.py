#!/usr/bin/env python3
"""Combine metric CSVs and merge on the `excerpt_id` column.

By default, this script merges per-excerpt data from:
- data/reference-based_metrics/aggregated_inclusion
- data/reference-free_metrics/clarity
- data/reference-free_metrics/immediate_relevance
- data/reference-free_metrics/specificity
- data/reference-free_metrics/attributed-meaning
- data/reference-free_metrics/self-reportedness
- data/reference-free_metrics/spontaneity
- data/reference-free_metrics/rq_relevance
- data/reference-free_metrics/length_and_surprisal
- data/reference-free_metrics/token_counts
- data/interviewer_techniques

Default output is written to data/analysis/initial_data.csv.
"""
from pathlib import Path
import argparse
import pandas as pd
import sys
import re


FOLDERS = [
    Path("data/reference-based_metrics/aggregated_inclusion"),
    Path("data/reference-free_metrics/clarity"),
    Path("data/reference-free_metrics/immediate_relevance"),
    Path("data/reference-free_metrics/specificity"),
    Path("data/reference-free_metrics/attributed-meaning"),
    Path("data/reference-free_metrics/self-reportedness"),
    Path("data/reference-free_metrics/spontaneity"),
    Path("data/reference-free_metrics/rq_relevance"),
    Path("data/reference-free_metrics/length_and_surprisal"),
    Path("data/reference-free_metrics/token_counts"),
    Path("data/interviewer_techniques")
]

QUOTE_ASSIGN_FOLDER = Path("data/paper_quotes/quote_assignments")


def read_and_concat_csvs(folder_path: Path) -> pd.DataFrame:
    csv_paths = sorted([p for p in folder_path.glob("*.csv") if p.is_file()])
    if not csv_paths:
        print(f"Warning: no CSVs found in {folder_path}")
        return pd.DataFrame()
    dataframe_list = []
    for csv_path in csv_paths:
        try:
            dataframe = pd.read_csv(csv_path)
            dataframe_list.append(dataframe)
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")
    if not dataframe_list:
        return pd.DataFrame()
    return pd.concat(dataframe_list, ignore_index=True)


# No prefixing: keep original column names. When merging, duplicate
# non-key columns will be dropped (keep first occurrence).


def main(output_csv_path: Path, join_type: str, include_quotes: bool = False):
    combined_dfs = []
    for folder in FOLDERS:
        print(f"Reading folder: {folder}")
        df = read_and_concat_csvs(folder)
        if df.empty:
            print(f"Skipping empty folder: {folder}")
            continue
        if "excerpt_id" not in df.columns:
            print(f"Warning: 'excerpt_id' column not found in files from {folder}. Skipping.")
            continue
        df = df.drop_duplicates(subset=["excerpt_id"]).reset_index(drop=True)
        # Rename gpt5_judgement to the appropriate metric name when present
        if "gpt5_judgement" in df.columns:
            if folder.name == "clarity":
                df = df.rename(columns={"gpt5_judgement": "clarity"})
            elif folder.name == "immediate_relevance":
                df = df.rename(columns={"gpt5_judgement": "immediate_relevance"})
            elif folder.name == "specificity":
                df = df.rename(columns={"gpt5_judgement": "specificity"})
            elif folder.name == "attributed-meaning":
                df = df.rename(columns={"gpt5_judgement": "attributed_meaning"})
            elif folder.name == "self-reportedness":
                df = df.rename(columns={"gpt5_judgement": "self_reportedness"})
            elif folder.name == "spontaneity":
                df = df.rename(columns={"gpt5_judgement": "spontaneity"})
        if "interviewer_techniques" in folder.name:
            df = df.rename(columns={"gpt5_cat_1": "intro_context", "gpt5_cat_2":"support_rapport", "gpt5_cat_3":"elaboration", "gpt5_cat_4": "specifying",
                                    "gpt5_cat_5":"direct", "gpt5_cat_6":"indirect", "gpt5_cat_7": "structuring", "gpt5_cat_8":"interpreting", "gpt5_cat_9":"clarification"})

        # Special handling for rq_relevance: melt gpt5_judgement_# columns into long form
        if folder.name == "rq_relevance":
            # find columns like gpt5_judgement_1, gpt5_judgement1, etc.
            jq_cols = [c for c in df.columns if re.match(r"^gpt5_judgement_?\d+$", c, flags=re.I)]
            if jq_cols:
                id_vars = [c for c in df.columns if c not in jq_cols]
                df_long = pd.melt(df, id_vars=id_vars, value_vars=jq_cols, var_name="rq_var", value_name="rq_relevance")
                # extract rq number and coerce to integer
                df_long["rq"] = df_long["rq_var"].str.extract(r"(\d+)$", flags=re.I)[0]
                df_long["rq"] = pd.to_numeric(df_long["rq"], errors="coerce").astype("Int64")
                df_long = df_long.drop(columns=["rq_var"]) 
                # drop rows without a rq relevance judgement
                df_long = df_long.dropna(subset=["rq_relevance"]).reset_index(drop=True)
                df = df_long
        # Special handling for surprisal keep only requested numeric cols
        if 'length_and_surprisal' in folder.parts:
            # keep excerpt_id and requested metrics only
            keep = [c for c in ['excerpt_id', 'n_tokens', 'perplexity', 'total_surprisal'] if c in df.columns]
            if 'excerpt_id' in keep:
                df = df[keep].drop_duplicates(subset=['excerpt_id']).reset_index(drop=True)
        combined_dfs.append(df)

    if not combined_dfs:
        print("No dataframes to merge. Exiting.")
        sys.exit(1)

    # Ensure output folder exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Separate dataframes that are 'rq'-level (have an 'rq' column) from those that are not.
    dfs_with_rq = [df for df in combined_dfs if 'rq' in df.columns]
    dfs_no_rq = [df for df in combined_dfs if 'rq' not in df.columns]

    if not dfs_no_rq and dfs_with_rq:
        # If all inputs are rq-level, start from the first rq-level df
        merged = dfs_with_rq.pop(0)
    else:
        # Merge all non-rq dataframes on 'excerpt_id' first
        merged = dfs_no_rq[0]
        for df in dfs_no_rq[1:]:
            merged = pd.merge(merged, df, on="excerpt_id", how=join_type, suffixes=("", "_dup"))
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            if dup_cols:
                merged = merged.drop(columns=dup_cols)

    # Optionally pivot avg_rq* and max_rq* into long format
    # Identify avg and max rq columns (supports both `avg_rq1` and `avg_rq_1` forms)
    def rq_cols(prefix):
        pattern = re.compile(rf"^{prefix}_rq_?\d+$", flags=re.I)
        return [c for c in merged.columns if pattern.match(c)]

    avg_cols = rq_cols("avg")
    max_cols = rq_cols("max")
    nc_cols = rq_cols("nc")

    # If user wants pivoting, do it after merging
    if getattr(main, "pivot_rq", False):
        # exclude avg/max/nc rq columns from id_vars so melts work correctly
        id_vars = [c for c in merged.columns if c not in (avg_cols + max_cols + nc_cols)]
        if not avg_cols and not max_cols:
            print("No avg_rq* or max_rq* columns found to pivot.")
        else:
            # Melt avg
            if avg_cols:
                avg_long = pd.melt(merged, id_vars=id_vars, value_vars=avg_cols, var_name="rq_var", value_name="avg")
                avg_long["rq"] = avg_long["rq_var"].str.extract(r"rq_?(\d+)$", flags=re.I)
                avg_long["rq"] = pd.to_numeric(avg_long["rq"], errors="coerce").astype("Int64")
                avg_long = avg_long.drop(columns=["rq_var"])
            else:
                avg_long = None

            # Melt max
            if max_cols:
                max_long = pd.melt(merged, id_vars=id_vars, value_vars=max_cols, var_name="rq_var", value_name="max")
                max_long["rq"] = max_long["rq_var"].str.extract(r"rq_?(\d+)$", flags=re.I)
                max_long["rq"] = pd.to_numeric(max_long["rq"], errors="coerce").astype("Int64")
                max_long = max_long.drop(columns=["rq_var"])
            else:
                max_long = None

            # Melt nc
            if nc_cols:
                nc_long = pd.melt(merged, id_vars=id_vars, value_vars=nc_cols, var_name="rq_var", value_name="nc")
                nc_long["rq"] = nc_long["rq_var"].str.extract(r"rq_?(\d+)$", flags=re.I)
                nc_long["rq"] = pd.to_numeric(nc_long["rq"], errors="coerce").astype("Int64")
                nc_long = nc_long.drop(columns=["rq_var"])
            else:
                nc_long = None

            # Combine available long tables (avg/max/nc)
            longs = [df for df in (avg_long, max_long, nc_long) if df is not None]
            if longs:
                merged = longs[0]
                for lf in longs[1:]:
                    merged = pd.merge(merged, lf, on=id_vars + ["rq"], how=join_type, suffixes=("", "_dup"))
                    dup_cols = [c for c in merged.columns if c.endswith("_dup")]
                    if dup_cols:
                        merged = merged.drop(columns=dup_cols)
            else:
                merged = pd.DataFrame(columns=id_vars + ['rq'])

            # drop rows where all of avg, max, and nc are missing
            drop_subset = [c for c in ["avg", "max", "nc"] if c in merged.columns]
            if drop_subset:
                merged = merged.dropna(subset=drop_subset, how="all").reset_index(drop=True)

    # After pivoting, merge any rq-level dataframes (e.g., rq_relevance) on ['excerpt_id','rq']
    for df in dfs_with_rq:
        if 'rq' in df.columns:
            df['rq'] = pd.to_numeric(df['rq'], errors='coerce').astype('Int64')
            if 'rq' in merged.columns:
                merged = pd.merge(merged, df, on=['excerpt_id', 'rq'], how=join_type, suffixes=("", "_dup"))
            else:
                merged = pd.merge(merged, df.drop(columns=['rq']), on='excerpt_id', how=join_type, suffixes=("", "_dup"))
        else:
            merged = pd.merge(merged, df, on='excerpt_id', how=join_type, suffixes=("", "_dup"))
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        if dup_cols:
            merged = merged.drop(columns=dup_cols)

    # Quotes: optionally read quote assignments and mark excerpts that contain any utterance
    if include_quotes:
        qa_df = read_and_concat_csvs(QUOTE_ASSIGN_FOLDER)
        utterances = []
        if not qa_df.empty:
            utter_col = next((c for c in qa_df.columns if 'utter' in c.lower()), None)
            if utter_col:
                utterances = qa_df[utter_col].dropna().astype(str).str.strip().unique().tolist()

        if 'excerpt_id' in merged.columns:
            if utterances:
                lower_utts = [u.lower() for u in utterances if u]
                def check_quoted(ex):
                    if pd.isna(ex):
                        return 0
                    ex_low = str(ex).lower()
                    for u in lower_utts:
                        if u and u in ex_low:
                            return 1
                    return 0
                merged['quoted'] = merged['excerpt_id'].apply(check_quoted)
            else:
                merged['quoted'] = 0

    if isinstance(merged, pd.DataFrame) and 'question' in merged.columns:
        merged = merged.drop(columns=['question'])

    # Write out
    merged.to_csv(output_csv_path, index=False)
    print(f"Wrote merged CSV to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CSVs from folders and join on 'excerpt_id'.")
    parser.add_argument("--out", "-o", default="data/analysis/initial_data.csv", help="Output CSV path")
    parser.add_argument("--how", "-w", default="outer", choices=["left", "right", "inner", "outer"], help="Join type when merging on 'excerpt_id'")
    parser.add_argument("--pivot-rq", dest="pivot_rq", action="store_true", help="Pivot avg_rq* and max_rq* into long format with columns 'avg','max','rq'")
    parser.add_argument("--include-quotes", "-q", dest="include_quotes", action="store_true", help="Enable quote-assignment matching (reads data/paper_quotes/quote_assignments)")
    args = parser.parse_args()

    out_path = Path(args.out)
    # attach pivot flag to main for use inside
    setattr(main, "pivot_rq", args.pivot_rq)
    main(out_path, args.how, args.include_quotes)
