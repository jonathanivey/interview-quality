#!/usr/bin/env python3
"""Compute length, perplexity, and surprisal metrics for interview excerpts.

This script supports three modes:
- score a single text with `--text`
- score one text per line with `--text-file`
- batch-process excerpt CSVs from `--input-dir` and write per-file outputs

Usage examples:
    python reference-free_metrics/compute_length_and_surprisal.py --text "This is a test sentence."
    python reference-free_metrics/compute_length_and_surprisal.py --input-dir data/excerpt_transcripts --output-dir data/reference-free_metrics/length_and_surprisal

Install requirements:
    pip install --upgrade "transformers>=4.0.0" torch numpy

Notes:
- Uses a causal language model (AutoModelForCausalLM).
- Uses model-provided `loss` when available, else computes NLL from logits.
- Includes optional unigram surprisal based on token frequency counts.

Includes logic adapted from:
https://github.com/byungdoh/llm_surprisal/blob/eacl24/get_unigram_surprisal.py
"""

import argparse
import math
import sys
import os
import glob
import csv
from typing import Tuple, Optional, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


def _compute_perplexity_with_model(text: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, device: str, context: Optional[str] = None) -> Tuple[float, float, int]:
    """Compute perplexity for a single string using a preloaded model and tokenizer.

    If `context` is provided, the model will receive the context preceding the
    utterance, but the reported loss/perplexity will be computed only over the
    utterance tokens (the context tokens are ignored for the loss via labels
    set to -100). Returns a tuple: (perplexity, loss, n_tokens).
    """
    # Determine a safe maximum length for this model/tokenizer.
    try:
        model_max_len = int(tokenizer.model_max_length)
    except Exception:
        model_max_len = getattr(model.config, "max_position_embeddings", 1024)
    # Some tokenizers report extremely large model_max_length; fall back when unrealistic
    if model_max_len is None or model_max_len <= 0 or model_max_len > 10 ** 6:
        model_max_len = getattr(model.config, "max_position_embeddings", 1024)

    # Tokenize once without truncation to detect long inputs, then re-tokenize with truncation.
    if context:
        enc_full = tokenizer(context, text, return_tensors="pt")
    else:
        enc_full = tokenizer(text, return_tensors="pt")
    seq_len_full = int(enc_full["input_ids"].size(1))
    if seq_len_full > model_max_len:
        print(f"Warning: input token length {seq_len_full} > model max {model_max_len}; truncating input.", file=sys.stderr)

    if context:
        enc = tokenizer(context, text, return_tensors="pt", truncation=True, max_length=model_max_len)
        # compute prefix length (number of tokens coming from context)
        enc_context = tokenizer(context, return_tensors="pt", truncation=True, max_length=model_max_len)
        prefix_len = int(enc_context["input_ids"].size(1))
        # clamp prefix_len to the actual combined sequence length
        if prefix_len >= int(enc["input_ids"].size(1)):
            prefix_len = int(enc["input_ids"].size(1))
    else:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=model_max_len)
        prefix_len = 0
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Prepare labels such that context/prefix tokens are ignored (label == -100)
    labels = input_ids.clone()
    if prefix_len > 0:
        labels[:, :prefix_len] = -100

    # Preferred: ask model to compute loss directly (works for most causal LM implementations)
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if getattr(outputs, "loss", None) is not None:
            loss = float(outputs.loss)
            perplexity = math.exp(loss)
            # count only non-ignored tokens
            try:
                n_tokens = int((labels != -100).sum().item())
            except Exception:
                n_tokens = int(input_ids.size(1) - prefix_len)
            return perplexity, loss, n_tokens
    except Exception:
        # Fall through to manual computation below
        pass

    # Manual computation from logits (shifted tokens)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab)

        # shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        shift_labels = shift_labels.unsqueeze(-1)
        token_log_probs = log_probs.gather(-1, shift_labels).squeeze(-1)

        seq_len = input_ids.size(1)
        # build mask to include only utterance tokens (exclude context/prefix)
        positions = torch.arange(seq_len, device=device)
        shift_pos_mask = (positions[1:] >= prefix_len)

        if attention_mask is not None:
            shift_attn = attention_mask[:, 1:].contiguous()
            combined_mask = shift_attn & shift_pos_mask
            token_log_probs = token_log_probs * combined_mask
            n_tokens = int(combined_mask.sum().item())
        else:
            combined_mask = shift_pos_mask
            token_log_probs = token_log_probs * combined_mask
            n_tokens = int(combined_mask.sum().item())

        if n_tokens == 0:
            return float("inf"), 0.0, 0

        # negative log-likelihood (only over utterance tokens)
        nll = -float(token_log_probs.sum().item())
        loss = nll / max(1, n_tokens)
        perplexity = math.exp(loss)
        return perplexity, loss, n_tokens


def compute_perplexity(text: str, model_name: str = "gpt2", device: Optional[str] = None) -> Tuple[float, float, int]:
    """Backward-compatible wrapper that loads the model/tokenizer and calls the helper.

    Returns (perplexity, loss, n_tokens).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model '{model_name}' on device '{device}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure there's a pad_token for some tokenizers
    if tokenizer.pad_token is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(model_name)
    # If tokenizer was expanded (e.g., we added a pad token), resize model embeddings.
    try:
        tok_size = len(tokenizer)
        emb_size = model.get_input_embeddings().weight.size(0)
        if tok_size != emb_size:
            model.resize_token_embeddings(tok_size)
    except Exception:
        # If anything goes wrong here, proceed without resizing; caller will see errors if necessary.
        pass

    model.to(device)
    model.eval()

    return _compute_perplexity_with_model(text, tokenizer, model, device)


def compute_surprisal_for_text(text: str, tokenizer: AutoTokenizer, counts: np.ndarray, mode: str = "word"):
    """Compute token- or word-level surprisal for `text` using unigram `counts`.

    Returns a list of (unit_str, surprisal_value) tuples where unit is token or word.
    """
    # prepare counts and totals
    counts_arr = counts.squeeze()
    log_total_counts = np.log2(np.sum(counts_arr))

    tok_out = tokenizer(text)
    ids = tok_out.get("input_ids")
    # tokenizer(text) sometimes returns nested lists
    if ids and isinstance(ids[0], list):
        ids = ids[0]

    toks = tokenizer.convert_ids_to_tokens(ids)

    # safe lookup for counts (use 1 for unknown ids to avoid log2(0))
    safe_counts = np.array([counts_arr[i] if 0 <= i < len(counts_arr) else 1 for i in ids], dtype=np.float64)
    # avoid zeros
    safe_counts[safe_counts <= 0] = 1.0
    surp = log_total_counts - np.log2(safe_counts)

    if mode == "token":
        out = []
        for i in range(len(toks)):
            cleaned_tok = tokenizer.convert_tokens_to_string([toks[i]]).replace(" ", "")
            out.append((cleaned_tok, float(surp[i])))
        return out

    # word-level: aggregate token surprisals into words using whitespace split
    words = text.split(" ")
    out = []
    curr_word_surp = []
    curr_toks = []
    curr_word_ix = 0
    for i in range(len(toks)):
        curr_word_surp.append(float(surp[i]))
        curr_toks += [toks[i]]
        curr_toks_str = tokenizer.convert_tokens_to_string(curr_toks)
        # when the token-joined string matches the next whitespace-split word, emit
        if curr_word_ix < len(words) and words[curr_word_ix] == curr_toks_str.strip():
            out.append((curr_toks_str.strip(), sum(curr_word_surp)))
            curr_word_surp = []
            curr_toks = []
            curr_word_ix += 1

    # if leftover tokens (mismatch between whitespace split and tokenizer), flush them as a single unit
    if curr_toks:
        out.append((tokenizer.convert_tokens_to_string(curr_toks).strip(), sum(curr_word_surp)))
    return out


def _parse_args():
    parser = argparse.ArgumentParser(description="Compute perplexity of a sentence using Hugging Face models")
    parser.add_argument("--text", type=str, help="Text string to measure perplexity for")
    parser.add_argument("--text-file", type=str, help="File with one sentence per line to measure (prints each)")
    parser.add_argument("--input-dir", type=str, default="data/excerpt_transcripts", help="Directory with CSV files to process (default: data/excerpt_transcripts)")
    parser.add_argument("--output-dir", type=str, default="data/reference-free_metrics/perplexity", help="Directory to write output CSV")
    parser.add_argument("--model", type=str, default="gpt2", help="Hugging Face model name (default: gpt2)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (default: auto detect CUDA)")
    parser.add_argument("--subfolder", type=str, default="gpt2", help="Subfolder to output filenames (default: none)")
    parser.add_argument("--surprisal-mode", type=str, choices=["none", "token", "word"], default="word", help="Compute unigram surprisal: 'none' (default), 'token', or 'word'")
    parser.add_argument("--counts-file", type=str, default="data/resources/the_pile_16k_unigrams.npy", help="Path to unigram counts .npy file (default: data/the_pile_16k_unigrams.npy)")
    # Per-file outputs are always written (no flag).
    return parser.parse_args()


def main():
    args = _parse_args()
    # load counts if surprisal requested
    counts = None
    surprisal_tokenizer = None
    if getattr(args, "surprisal_mode", "none") != "none":
        try:
            counts = np.load(args.counts_file)
        except Exception as e:
            print(f"Warning: could not load counts file '{args.counts_file}': {e}", file=sys.stderr)
            counts = None
        try:
            surprisal_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", revision="step143000")
        except Exception as e:
            print(f"Warning: could not load surprisal tokenizer: {e}", file=sys.stderr)
            surprisal_tokenizer = None
    # If a single text or text-file is provided, keep original behavior
    if args.text_file:
        try:
            with open(args.text_file, "r", encoding="utf-8") as fh:
                lines = [l.strip() for l in fh if l.strip()]
        except Exception as e:
            print(f"Error opening file: {e}", file=sys.stderr)
            sys.exit(2)

        # load model once
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model = AutoModelForCausalLM.from_pretrained(args.model)
        # Resize embeddings if tokenizer changed
        try:
            tok_size = len(tokenizer)
            emb_size = model.get_input_embeddings().weight.size(0)
            if tok_size != emb_size:
                model.resize_token_embeddings(tok_size)
        except Exception:
            pass
        model.to(device)
        model.eval()

        for i, line in enumerate(lines, start=1):
            ppl, loss, n_tokens = _compute_perplexity_with_model(line, tokenizer, model, device)
            print(f"[{i}] tokens={n_tokens} loss={loss:.4f} ppl={ppl:.4f}  text={line}")
            if getattr(args, "surprisal_mode", "none") != "none" and counts is not None and surprisal_tokenizer is not None:
                surp = compute_surprisal_for_text(line, surprisal_tokenizer, counts, mode=args.surprisal_mode)
                total = sum(v for _, v in surp)
                print(f"total_surprisal={total:.6f}")
        return

    if args.text:
        ppl, loss, n_tokens = compute_perplexity(args.text, model_name=args.model, device=args.device)
        print(f"tokens={n_tokens} loss={loss:.4f} perplexity={ppl:.4f}")
        if getattr(args, "surprisal_mode", "none") != "none" and counts is not None and surprisal_tokenizer is not None:
            surp = compute_surprisal_for_text(args.text, surprisal_tokenizer, counts, mode=args.surprisal_mode)
            total = sum(v for _, v in surp)
            print(f"total_surprisal={total:.6f}")
        return

    # Otherwise: process all CSVs in the input directory
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_paths:
        print(f"No CSV files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(args.model)
    # Resize embeddings when tokenizer vocabulary changed
    try:
        tok_size = len(tokenizer)
        emb_size = model.get_input_embeddings().weight.size(0)
        if tok_size != emb_size:
            model.resize_token_embeddings(tok_size)
    except Exception:
        pass
    model.to(device)
    model.eval()

    for path in csv_paths:
        basename = os.path.basename(path)
        per_results: List[Tuple[str, str, Optional[str], int, float, float, Optional[float]]] = []
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                # find columns case-insensitively
                fieldmap = {name.lower(): name for name in reader.fieldnames or []}
                if "excerpt" not in fieldmap:
                    print(f"Skipping {basename}: no 'excerpt' column", file=sys.stderr)
                    continue
                utter_col = fieldmap["excerpt"]
                # require an existing excerpt_id column in the input (no fallback)
                excerpt_id_col = fieldmap.get("excerpt_id")
                if not excerpt_id_col:
                    print(f"Skipping {basename}: no 'excerpt_id' column", file=sys.stderr)
                    continue
                # detect question column if present
                question_col = fieldmap.get("question")
                for i, row in enumerate(reader, start=1):
                    text = (row.get(utter_col) or "").strip()
                    if not text:
                        continue
                    question = None
                    if question_col:
                        question = (row.get(question_col) or "").strip()
                        if not question:
                            question = None
                    ppl, loss, n_tokens = _compute_perplexity_with_model(text, tokenizer, model, device, context=question)
                    # compute total surprisal per excerpt if requested and counts available
                    total_surprisal = None
                    if getattr(args, "surprisal_mode", "none") != "none" and counts is not None and surprisal_tokenizer is not None:
                        try:
                            surp = compute_surprisal_for_text(text, surprisal_tokenizer, counts, mode=args.surprisal_mode)
                            total_surprisal = float(sum(v for _, v in surp))
                        except Exception:
                            total_surprisal = None
                    # use provided excerpt_id; skip row if missing or empty
                    excerpt_id_val = (row.get(excerpt_id_col) or "").strip()
                    if not excerpt_id_val:
                        continue
                    per_results.append((excerpt_id_val, text, question, n_tokens, loss, ppl, total_surprisal))
        except Exception as e:
            print(f"Error processing {basename}: {e}", file=sys.stderr)
        # Always write a per-file output for this input CSV (strip .csv suffix)
        if per_results:
            name_root, ext = os.path.splitext(basename)
            per_out = os.path.join(output_dir, args.subfolder, f"{name_root}.csv") if args.subfolder else os.path.join(output_dir, f"{name_root}.csv")
            try:
                with open(per_out, "w", newline="", encoding="utf-8") as pf:
                    writer = csv.writer(pf)
                    writer.writerow(["excerpt_id", "excerpt", "question", "n_tokens", "loss", "perplexity", "total_surprisal"])
                    for excerpt_id, utt, question, n_tokens, loss, ppl, total_surprisal in per_results:
                        s_val = f"{total_surprisal:.6f}" if total_surprisal is not None else ""
                        writer.writerow([excerpt_id, utt, question or "", n_tokens, f"{loss:.6f}", f"{ppl:.6f}", s_val])
                print(f"Wrote {len(per_results)} rows to {per_out}")
            except Exception as e:
                print(f"Error writing per-file output for {basename}: {e}", file=sys.stderr)

    # Completed processing all files. Per-file outputs have been written above.


if __name__ == "__main__":
    main()
