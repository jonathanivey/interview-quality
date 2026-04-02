import pdfplumber
import re
import csv
import os

# define input and output directories
in_directory = "data/raw_data/doi-10.5064-f6ymwpux"
out_individual = "data/processed_transcripts/doi-10.5064-f6ymwpux"
out_combined = "data/combined_transcripts/doi-10.5064-f6ymwpux__combined_transcripts.csv"

def process_pdf(input_pdf, output_csv):
    # extract text from PDF and mark bolded words as speaker tags
    transcript = ""
    with pdfplumber.open(input_pdf) as pdf:
        for page in pdf.pages:
            # Get page width and height and restrict to main body to avoid headers/footers
            width = page.width
            height = page.height
            bbox = (0, 70, width, height - 70)
            page_crop = page.within_bbox(bbox)

            # Try to extract words with font attributes; fall back if unsupported
            try:
                words = page_crop.extract_words(extra_attrs=["fontname"])
            except TypeError:
                words = page_crop.extract_words()

            page_text = []
            for w in words:
                word = w.get('text', '')
                fontname = (w.get('fontname') or '')
                is_bold = False
                if fontname:
                    fn = fontname.lower()
                    if 'bold' in fn or fn.endswith('bd') or fn.endswith('-bd') or fn.endswith('b'):
                        is_bold = True

                if is_bold:
                    page_text.append(f"<<B:{word}>>")
                else:
                    page_text.append(word)

            # join words preserving spacing
            transcript += ' '.join(page_text) + '\n'

    # Split transcript on bold tags and assign utterances
    parts = re.split(r"<<B:(.*?)>>", transcript, flags=re.S)
    rows = []
    # parts: [pre_text, tag1, text1, tag2, text2, ...]
    for i in range(1, len(parts), 2):
        tag = parts[i].strip()
        content = parts[i+1] if i+1 < len(parts) else ''
        if not tag:
            continue
        first = tag[0].lower()
        if first == 'i':
            speaker = 'interviewer'
        elif first == 'p':
            speaker = 'participant'
        else:
            # skip tags that don't start with I/i or P/p
            continue

        text = re.sub(r"\s+", ' ', content).strip()
        # drop leading colons if present
        if text.startswith(':'):
            text = text.lstrip(':').strip()
        rows.append([speaker, text])

    # write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker", "utterance"])
        writer.writerows(rows)

# list of file names that contain "Interview_CG" or "Interview_Patient"
def collect_input_pdf_filenames(directory):
    interview_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "_interview" in file:
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
                participant_name = source_name.split('_')[2]  # Extract participant identifier
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    next(csv_reader)  # Skip the header row
                    
                    i=0
                    # Read and add source column to each row
                    for row in csv_reader:
                        row.append(source_name)
                        row.append(participant_name)  # Add participant name
                        row.append(source_name + '_' + str(i))
                        combined_data.append(row)
                        i+=1

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
