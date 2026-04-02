import pdfplumber
import re
import csv
import os
from collections import defaultdict


def is_italic_font(fontname: str) -> bool:
    if not fontname:
        return False
    return bool(re.search(r"italic|oblique|it\b|ital|slant", fontname, flags=re.I))

# define input and output directories
in_directory = "data/raw_data/doi-10.5064-f6zp448b/QDR Project 10085_O.Neill"
out_individual = "data/processed_transcripts/doi-10.5064-f6zp448b"
out_combined = "data/combined_transcripts/doi-10.5064-f6zp448b__combined_transcripts.csv"
# number of points (PDF coordinate units) to trim from top of each page
# increase if headers run deeper into the page; set to 0 to disable trimming
TOP_CROP = 70


def process_pdf(input_pdf, output_csv):
    # Extract text from PDF by inspecting character fonts.
    # Runs of italic text -> interviewer; non-italic -> participant.
    rows = []
    with pdfplumber.open(input_pdf) as pdf:
        for page in pdf.pages:
            # Get page characters and approximate reading order by line ('top') then x0
            # Filter out characters that fall in the top cropped region
            all_chars = page.chars
            if all_chars is None:
                chars = []
            else:
                chars = [c for c in all_chars if c.get('top', 0) >= TOP_CROP]
            if not chars:
                # Fallback: treat whole page text (below crop) as participant
                try:
                    text = (page.within_bbox((0, TOP_CROP, page.width, page.height)).extract_text() or "").strip().replace('\n', ' ')
                except Exception:
                    text = (page.extract_text() or "").strip().replace('\n', ' ')
                if text:
                    rows.append(["participant", text])
                continue

            # Group characters by rounded vertical position to form lines
            lines = defaultdict(list)
            for ch in chars:
                top_key = round(ch.get('top', 0))
                lines[top_key].append(ch)

            # Process lines in top-to-bottom order
            ordered_tops = sorted(lines.keys())
            # Build runs across the page preserving order
            for top in ordered_tops:
                line_chars = sorted(lines[top], key=lambda c: c.get('x0', 0))
                run_text = []
                run_italic = None
                prev_char = None
                for ch in line_chars:
                    fontname = ch.get('fontname') or ch.get('font') or ''
                    italic = is_italic_font(fontname)

                    # Insert space if there's a substantial gap from previous char
                    if prev_char is not None:
                        gap = ch.get('x0', 0) - prev_char.get('x1', 0)
                        if gap > (ch.get('size', 1) * 0.5):
                            run_text.append(' ')

                    if run_italic is None:
                        run_italic = italic

                    if italic == run_italic:
                        run_text.append(ch.get('text', ''))
                    else:
                        # flush previous run
                        text_run = ''.join(run_text).strip()
                        if text_run:
                            speaker = 'interviewer' if run_italic else 'participant'
                            rows.append([speaker, text_run.replace('\n', ' ')])
                        # start new run
                        run_text = [ch.get('text', '')]
                        run_italic = italic

                    prev_char = ch

                # flush end-of-line run
                if run_text:
                    text_run = ''.join(run_text).strip()
                    if text_run:
                        speaker = 'interviewer' if run_italic else 'participant'
                        rows.append([speaker, text_run.replace('\n', ' ')])
    

    # Ensure output directory exists
    # drop the very first extracted line from this file (commonly a header)
    if rows:
        rows = rows[1:]

    # merge consecutive utterances by the same speaker
    merged = []
    for speaker, text in rows:
        text = (text or '').strip()
        if not text:
            continue
        if not merged or merged[-1][0] != speaker:
            merged.append([speaker, text])
        else:
            merged[-1][1] = merged[-1][1] + ' ' + text

    rows = merged

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker", "utterance"])
        writer.writerows(rows)

# list of file names
def collect_input_pdf_filenames(directory):
    interview_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "_A_" in file:
                interview_files.append(file)
    return interview_files


# extract file names
input_pdf_files = collect_input_pdf_filenames(in_directory)

# Function to combine all CSV files
def write_combined_transcript_csv(directory, output_file):
    combined_data = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                source_name = file.replace('.csv', '')  # Remove .csv extension for source identifier
                participant_name = source_name.split('_')[-1]  # Extract participant identifier
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    next(csv_reader)  # Skip the header row
                    i = 0
                    # Read and add source column to each row
                    for row in csv_reader:
                        row.append(source_name)
                        row.append(participant_name)  # Add participant name
                        row.append(source_name + '_' + str(i))
                        combined_data.append(row)
                        i += 1

    # Write combined data to new CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['speaker', 'utterance', 'source', 'participant', 'utterance_id'])  # Write header with new source column
        writer.writerows(combined_data)

# process each file
for file in input_pdf_files:
    input_pdf = os.path.join(in_directory, file)
    output_csv = os.path.join(out_individual, file.replace(".pdf", ".csv"))
    process_pdf(input_pdf, output_csv)

# Combine all CSV files into one
write_combined_transcript_csv(out_individual, out_combined)

