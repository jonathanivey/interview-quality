import pdfplumber
import re
import csv
import os

# define input and output directories
in_directory = "data/raw_data/doi-10.5064-f68tojjy"
out_individual = "data/processed_transcripts/doi-10.5064-f68tojjy"
out_combined = "data/combined_transcripts/doi-10.5064-f68tojjy__combined_transcripts.csv"


def process_pdf(input_pdf, output_csv):
    # extract text from PDF (exclude headers/footers)
    transcript = ""
    with pdfplumber.open(input_pdf) as pdf:
        for page in pdf.pages:
            width = page.width
            height = page.height
            bbox = (0, 0, width, height-60)
            text = page.within_bbox(bbox).extract_text()
            if text:
                transcript += "\n" + text

    if not transcript:
        return

    # normalize newlines
    transcript = transcript.replace('\r\n', '\n').replace('\r', '\n')

    # Find all timestamp markers and split into blocks
    ts_pattern = re.compile(r"\d+:\d+:\d+(?:\.\d+)?\s*-->\s*\d+:\d+:\d+(?:\.\d+)?", re.MULTILINE)
    matches = list(ts_pattern.finditer(transcript))

    rows = []
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(transcript)
        block = transcript[start:end].strip()
        if not block:
            continue

        # Collapse consecutive blank lines and split
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        # First non-empty line should be speaker, often in angle brackets
        speaker_line = lines[0]
        speaker = speaker_line
        m_sp = re.match(r"^<\s*(.*?)\s*>$", speaker_line)
        if m_sp:
            speaker = m_sp.group(1).strip()

        # Remaining lines are the utterance (may be one or many lines; may also be enclosed in <> )
        utter_lines = lines[1:]
        utter_text = ''
        if utter_lines:
            utter_text = '\n'.join(utter_lines).strip()
            # strip surrounding angle brackets if present across the joined text
            if utter_text.startswith('<') and utter_text.endswith('>'):
                utter_text = re.sub(r"^<\s*|\s*>$", '', utter_text).strip()

        # Label speaker: if name starts with 'I' or 'i' then interviewer
        label = 'interviewer' if speaker and speaker[0].lower() == 'i' else 'participant'

        utter_text = utter_text.replace('\n', ' ').strip()
        rows.append([label, utter_text])

    # ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker", "utterance"])
        writer.writerows(rows)


# list of file names that contain "_FR"
def collect_input_pdf_filenames(directory):
    interview_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "_FR" in file:
                interview_files.append(file)
    return interview_files


# extract file names
input_pdf_files = collect_input_pdf_filenames(in_directory)


# Function to combine all CSV files
def write_combined_transcript_csv(directory, output_file):
    combined_data = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                source_name = file.replace('.csv', '')
                participant_name = source_name.split('_')[3]

                with open(file_path, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    try:
                        next(csv_reader)
                    except StopIteration:
                        continue

                    i = 0
                    for row in csv_reader:
                        row.append(source_name)
                        row.append(participant_name)
                        row.append(source_name + '_' + str(i))
                        combined_data.append(row)
                        i += 1

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['speaker', 'utterance', 'source', 'participant', 'utterance_id'])
        writer.writerows(combined_data)


# process each file
for file in input_pdf_files:
    input_pdf = os.path.join(in_directory, file)
    output_csv = os.path.join(out_individual, file.replace('.pdf', '.csv'))
    process_pdf(input_pdf, output_csv)


# Combine all CSV files into one
write_combined_transcript_csv(out_individual, out_combined)
