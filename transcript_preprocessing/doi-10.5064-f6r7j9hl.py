import pdfplumber
import re
import csv
import os

# define input and output directories
in_directory = "data/raw_data/doi-10.5064-f6r7j9hl/Phase-1"
out_individual = "data/processed_transcripts/doi-10.5064-f6r7j9hl"
out_combined = "data/combined_transcripts/doi-10.5064-f6r7j9hl__combined_transcripts.csv"



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
            bbox = (0, 60, width, height - 60)
            text = page.within_bbox(bbox).extract_text()
            if text:
                transcript += text

    # Extract interviewer and interviewee utterances with regex
    pattern = r"(Speaker\s*\d+)\s*:?\s*(.*?)(?=(?:Speaker\s*\d+\s*:?\s*|$))"
    matches = re.findall(pattern, transcript, flags=re.S)

    rows = []
    for match in matches:
        speaker = match[0].strip().replace('Speaker 1', 'interviewer').replace('Speaker 2', 'participant').replace('Speaker 3', 'interviewer')
        text = match[1].strip().replace('\n', ' ')
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
            if "transcript" in file and "P09" not in file:
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


def remove_utterances_from_participants(removal_dict):
    """
    Remove the first N utterances from participant transcripts.
    
    Args:
        removal_dict: A dictionary mapping participant IDs (P01-P20) to number of utterances to remove (2 or 3)
                     Example: {'P01': 2, 'P05': 3, 'P12': 2}
    """
    for participant_id, num_utterances in removal_dict.items():
        # Construct the CSV file path for this participant
        csv_file = os.path.join(out_individual, f"Bezabih-Smith_anonymized_transcript_{participant_id}.csv")
        
        # Check if file exists
        if not os.path.exists(csv_file):
            print(f"Warning: File not found for {participant_id}: {csv_file}")
            continue
        
        # Read the CSV file
        rows = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)  # Keep the header
            rows.append(header)
            
            # Read all data rows
            for row in csv_reader:
                rows.append(row)
        
        # Remove the first N participant utterances (filter for "participant" speaker)
        removed_count = 0
        filtered_rows = [rows[0]]  # Start with header
        
        for i in range(1, len(rows)):
            if removed_count < num_utterances:
                removed_count += 1
                continue  # Skip this utterance
            filtered_rows.append(rows[i])
        
        # Write back to the CSV file
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(filtered_rows)


removal_dict = {
    'P01': 2, 'P02': 3, 'P03': 2, 'P04': 2, 'P05': 3, 'P06': 2, 'P07': 2, 'P08': 2, 'P09': 2,
    'P10': 3, 'P11': 2, 'P12': 2, 'P13': 2, 'P14': 3, 'P15': 3, 'P16': 2, 'P17': 2, 'P18': 2,
    'P19': 2, 'P20': 2, 'P21': 2, 'P22': 3
}
remove_utterances_from_participants(removal_dict)

# Combine all CSV files into one
write_combined_transcript_csv(out_individual, out_combined)