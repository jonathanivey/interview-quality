import pdfplumber
import re
import csv
import os

# define input and output directories
in_directory = "data/raw_data/doi-10.5064-f6agwujg/Interviews"
out_individual = "data/processed_transcripts/doi-10.5064-f6agwujg"
out_combined = "data/combined_transcripts/doi-10.5064-f6agwujg__combined_transcripts.csv"



def process_pdf(input_pdf, output_csv):
    # extract text from PDF (exclude headers/footers)
    transcript = ""
    with pdfplumber.open(input_pdf) as pdf:
        for page in pdf.pages:
            # Get page width and height
            width = page.width
            height = page.height

            # Define bounding box: (x0, top, x1, bottom)
            # Adjust these values as needed to exclude header/footer
            bbox = (0, 0, width, height-50)
            text = page.within_bbox(bbox).extract_text()
            if text:
                transcript += text

    # Extract interviewer and interviewee utterances
    # Match interviewer tags that are exactly 'I' and participant tags
    # that start with 'P' and may contain any characters before the ':' (e.g., 'P1:', 'P_12:', 'P (participant):')
    pattern = r"((?:I)|P[^:]*):\s*(.*?)(?=(?:I:|P[^:]*:)|$)"
    matches = re.finditer(pattern, transcript, flags=re.S)

    rows = []
    for m in matches:
        speaker_label = m.group(1)
        speaker = 'interviewer' if speaker_label == 'I' else 'participant'
        text = m.group(2).strip().replace('\n', ' ')
        rows.append([speaker, text])
    

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
            if "Interview_P" in file:
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

