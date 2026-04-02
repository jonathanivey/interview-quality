from openai import OpenAI
import argparse
import csv
import json
import re
import sys
import tempfile
from pathlib import Path
from time import sleep
import traceback

client = OpenAI()

# Increase CSV field size limit to handle very large fields that exceed the
# default parser limit (prevents "field larger than field limit" errors).
try:
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
except Exception as e:
    print(f"Warning: could not set csv.field_size_limit: {e}", file=sys.stderr)


def _canonical_stem(stem: str) -> str:
    return stem.split("__", 1)[0]


def _resolve_companion_file(directory: Path, stem: str, suffix: str) -> Path:
    exact = directory / f"{stem}{suffix}"
    if exact.exists():
        return exact

    base = _canonical_stem(stem)
    base_exact = directory / f"{base}{suffix}"
    if base_exact.exists():
        return base_exact

    candidates = sorted(p for p in directory.glob(f"{base}__*{suffix}") if p.is_file())
    if candidates:
        return sorted(candidates, key=lambda p: (len(p.name), p.name))[0]

    return exact

def build_prompt(excerpt: str, previous_excerpt: str = None, context_blurb: str = None) -> str:
    if previous_excerpt:
        previous = previous_excerpt
    else:
        previous = "N/A (this is the first excerpt in the transcript or previous excerpt context is unavailable)"

    return f"""You are an expert qualitative researcher analyzing interview data.

Rate the self-reportedness of the participant statement from in the current interview excerpt below on a scale from 1 to 3. In addition to the current excerpt, you are also provided with a short context blurb and the interview excerpt that immediately preceded the current excerpt in the transcript. These two sections are only to understand the context of the current excerpt, and your rating should be for participant statement in the current excerpt.

Scoring Rubric:
1. The participant statement is not interpretable without additional context.
2. The core idea of the participant statement is understandable but may require additional context for full understanding.
3. The participant statement is fully self-contained, not requiring any additional context to be interpretable.

CONTEXT BLURB (context only):
{context_blurb}

PREVIOUS INTERVIEW EXCERPT (context_only):
{previous}

CURRENT INTERVIEW EXCERPT (rate this):
{excerpt}


CRITICAL: Output only a single digit (1, 2, or 3). Do not write any additional text.

"""


def judge_row(excerpt: str, model: str = "gpt-5.4", max_retries: int = 1) -> str:
    prompt = build_prompt(excerpt)
    attempt = 0
    while attempt < max_retries:
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=16
            )
            # The old helper used resp.output_text
            text = getattr(resp, "output_text", None)
            if text is None:
                # fall back to parsing choices or output
                try:
                    # some client versions return resp.output[0].content[0].text
                    text = resp.output[0]["content"][0]["text"]
                except Exception:
                    text = str(resp)

            text = text.strip()
            m = re.search(r"[1-3]", text)
            if m:
                return m.group(0)
            return ""
        except Exception as e:
            attempt += 1
            wait = 1.0 * attempt
            print(f"API error (attempt {attempt}) - retrying in {wait}s: {e}", file=sys.stderr)
            if attempt >= max_retries:
                print("Max retries reached. Returning empty judgement.", file=sys.stderr)
                traceback.print_exc()
                return ""
            sleep(wait)


def _build_batch_request_line(custom_id: str, prompt: str, model: str, max_output_tokens: int = 16) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        },
    }


def _extract_output_text_from_response_obj(resp_obj: dict) -> str:
    # Robust extraction supporting multiple possible response shapes
    if resp_obj is None:
        return ""

    # If it's already a string
    if isinstance(resp_obj, str):
        return resp_obj.strip()

    if isinstance(resp_obj, dict):
        if isinstance(resp_obj.get("output_text"), str) and resp_obj["output_text"].strip():
            return resp_obj["output_text"].strip()

        if "response" in resp_obj and resp_obj["response"]:
            return _extract_output_text_from_response_obj(resp_obj["response"])
        if "result" in resp_obj and resp_obj["result"]:
            return _extract_output_text_from_response_obj(resp_obj["result"])

        try:
            out = resp_obj.get("output")
            if isinstance(out, list) and out:
                c = out[0].get("content")
                if isinstance(c, list) and c:
                    for item in c:
                        t = item.get("text") if isinstance(item, dict) else None
                        if isinstance(t, str) and t.strip():
                            return t.strip()
        except Exception:
            pass

        try:
            choices = resp_obj.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message")
                if isinstance(msg, dict):
                    cont = msg.get("content")
                    if isinstance(cont, str) and cont.strip():
                        return cont.strip()
                    if isinstance(cont, list) and cont:
                        for item in cont:
                            if isinstance(item, str) and item.strip():
                                return item.strip()
                    if isinstance(cont, dict):
                        t = cont.get("text")
                        if isinstance(t, str) and t.strip():
                            return t.strip()
        except Exception:
            pass

        if isinstance(resp_obj.get("text"), str) and resp_obj["text"].strip():
            return resp_obj["text"].strip()

    try:
        return str(resp_obj).strip()
    except Exception:
        return ""


def _submit_batch_and_wait(request_lines: list, csv_name: str) -> dict:
    # Write JSONL to a temporary file
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for line in request_lines:
            tmp.write(json.dumps(line, ensure_ascii=False) + "\n")
        tmp_path = tmp.name

    # Upload as a batch input file
    with open(tmp_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")

    # Create batch job
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"source_csv": csv_name},
    )

    batch_id = getattr(batch, "id", None) or (batch.get("id") if isinstance(batch, dict) else None)
    print(f"Created batch {batch_id} with {len(request_lines)} requests")

    # Poll until finished
    while True:
        b = client.batches.retrieve(batch_id)
        status = getattr(b, "status", None) or (b.get("status") if isinstance(b, dict) else None)
        print(f"Batch {batch_id} status: {status}")
        if status in {"completed", "failed", "cancelled", "canceled", "expired"}:
            batch = b
            break
        sleep(5)

    # Gather output file ids
    output_ids = []
    for attr in ("output_file_id", "output_file_ids"):
        val = getattr(batch, attr, None) if not isinstance(batch, dict) else batch.get(attr)
        if not val:
            continue
        if isinstance(val, (list, tuple)):
            output_ids.extend(val)
        else:
            output_ids.append(val)

    results_by_custom_id = {}
    if not output_ids:
        print("No output files found for batch — returning empty results", file=sys.stderr)
        return results_by_custom_id

    # Download and parse outputs
    for fid in output_ids:
        try:
            file_content = client.files.content(fid)
            # Handle SDK variants (.text or .content/.read())
            if hasattr(file_content, "text"):
                text = file_content.text
            else:
                try:
                    text = file_content.content.decode("utf-8")
                except Exception:
                    text = file_content.read().decode("utf-8")

            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                custom_id = obj.get("custom_id", "")
                if obj.get("error"):
                    results_by_custom_id[custom_id] = ""
                    continue

                # The actual model response may be wrapped in several layers.
                # Prefer response.body, then body, then result/output, else fall back to the whole object
                response_obj = None
                if isinstance(obj.get("response"), dict):
                    response_obj = obj["response"].get("body") or obj["response"]
                elif isinstance(obj.get("body"), dict):
                    response_obj = obj["body"]
                else:
                    response_obj = obj.get("result") or obj.get("output") or obj

                out_text = _extract_output_text_from_response_obj(response_obj)
                m = re.search(r"[1-3]", out_text)
                results_by_custom_id[custom_id] = m.group(0) if m else ""
        except Exception as e:
            print(f"Failed to download/parse batch output file {fid}: {e}", file=sys.stderr)

    return results_by_custom_id


def process_all(input_dir: Path, output_dir: Path, model: str = "gpt-5.4", delay: float = 0.05, max_rows: int = None, only_csv: str = None, dry_run: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if only_csv:
        csv_files = [p for p in csv_files if p.name == only_csv]
    if not csv_files:
        print(f"No CSV files found in {input_dir}", file=sys.stderr)
        return

    for csv_file in csv_files:
        stem = csv_file.stem
        print(f"Processing {csv_file.name}")

        context_blurb = ""
        context_blurb_path = _resolve_companion_file(Path("data/context_blurbs"), stem, ".txt")
        if context_blurb_path.exists():
            try:
                context_blurb = context_blurb_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                print(f"Warning: failed reading context blurb {context_blurb_path}: {e}", file=sys.stderr)
        else:
            print(f"Warning: context blurb not found for {csv_file.name} at {context_blurb_path}", file=sys.stderr)

        # Read CSV rows
        with csv_file.open("r", encoding="utf-8", newline="") as inf:
            reader = csv.DictReader(inf)
            fieldnames = reader.fieldnames[:] if reader.fieldnames else []
            rows = list(reader)
            if max_rows is not None and max_rows > 0:
                rows = rows[:max_rows]

        # Verify required columns
        if not rows:
            print(f"No rows in {csv_file}", file=sys.stderr)

        # Determine excerpt column name
        try:
            excerpt_col = next(f for f in fieldnames if f.lower() == 'excerpt')
        except StopIteration:
            print(f"File {csv_file} missing required 'excerpt' column", file=sys.stderr)
            continue

        source_col = None
        for f in fieldnames:
            if f.lower() == "source":
                source_col = f
                break
        if source_col is None:
            print(f"Warning: file {csv_file} missing 'source' column; previous excerpt context will be omitted", file=sys.stderr)

        # prepare single output column for judgement
        section_cols = ["gpt5_judgement"]
        out_fieldnames = fieldnames + section_cols

        out_path = output_dir / csv_file.name
        # Build batch request lines: one request per row (excerpt only)
        request_lines = []
        for idx, row in enumerate(rows):
            excerpt = (row.get(excerpt_col) or "")
            previous_excerpt = None
            if source_col and idx > 0:
                prev_row = rows[idx - 1]
                current_source = row.get(source_col)
                prev_source = prev_row.get(source_col)
                if current_source and prev_source == current_source:
                    previous_excerpt = (prev_row.get(excerpt_col) or "")

            prompt = build_prompt(excerpt, previous_excerpt=previous_excerpt, context_blurb=context_blurb)
            custom_id = f"{stem}:{idx}"
            request_lines.append(_build_batch_request_line(custom_id, prompt, model))

        if not request_lines:
            print(f"No requests to submit for {csv_file}", file=sys.stderr)
            continue

        if dry_run:
            dry_run_dir = Path("data/dry_run_files") / Path(__file__).stem
            dry_run_dir.mkdir(parents=True, exist_ok=True)
            dry_run_path = dry_run_dir / f"{csv_file.stem}.jsonl"
            with dry_run_path.open("w", encoding="utf-8") as dryf:
                for req in request_lines:
                    dryf.write(json.dumps({
                        "custom_id": req.get("custom_id", ""),
                        "prompt": req.get("body", {}).get("input", ""),
                    }, ensure_ascii=False) + "\n")
            print(f"Dry run: wrote prompts to {dry_run_path}")
            continue

        # Submit a single batch for this CSV and wait for results
        results_by_custom_id = _submit_batch_and_wait(request_lines, csv_file.name)

        # Write output CSV with judgements filled in
        with out_path.open("w", encoding="utf-8", newline="") as outf:
            writer = csv.DictWriter(outf, fieldnames=out_fieldnames)
            writer.writeheader()

            for idx, row in enumerate(rows):
                colname = section_cols[0]
                custom_id = f"{stem}:{idx}"
                row[colname] = results_by_custom_id.get(custom_id, "")
                writer.writerow(row)

        print(f"Wrote output to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch LLM judge for interview excerpts (excerpt-only)")
    parser.add_argument("--input-dir", default="data/excerpt_transcripts", help="Directory with input CSVs")
    parser.add_argument("--output-dir", default="data/reference-free_metrics/self-reportedness", help="Directory to write outputs")
    parser.add_argument("--model", default="gpt-5.4", help="Model to call")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay (s) between API calls")
    parser.add_argument("--max-rows", type=int, default=None, help="Process only the first N rows of each file")
    parser.add_argument("--only-csv", default=None, help="Process only the specified CSV filename (basename). Example: doi-10.5064-f6r7j9hl.csv")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and write them to data/dry_run_files instead of calling the API")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(2)
    process_all(input_dir, output_dir, model=args.model, delay=args.delay, max_rows=args.max_rows, only_csv=args.only_csv, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
