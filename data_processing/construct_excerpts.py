"""Assign participant utterances to the preceding interviewer question.

Reads all CSV files in `data/combined_transcripts/` and for each input file
produces an output CSV in `data/excerpt_transcripts/` with grouped excerpts.
Each excerpt starts at the most recent interviewer utterance that contains
more than four words and includes subsequent interviewer/participant lines
until the next qualifying interviewer utterance.

Output columns:
- `excerpt_id`: running excerpt id
- `excerpt`: combined transcript lines for the excerpt
- `source`: taken from the input rows (first non-empty source seen)
- `participant`: participant id(s) from the grouped utterances (joined with '|')
- `question`: the interviewer utterance that started the excerpt

Usage:
	python data_processing/construct_excerpts.py
	python data_processing/construct_excerpts.py --input-dir data/combined_transcripts
"""

import csv
import os
import argparse


def canonical_output_basenames(source_basename):
	"""Return canonical excerpt and mapping basenames for an input filename."""
	root, ext = os.path.splitext(source_basename)
	if root.endswith('__combined_transcripts'):
		base = root[:-len('__combined_transcripts')]
	elif root.endswith('__excerpt_transcripts'):
		base = root[:-len('__excerpt_transcripts')]
	elif root.endswith('__utterance_excerpt_pairs'):
		base = root[:-len('__utterance_excerpt_pairs')]
	else:
		base = root
	return f'{base}__excerpt_transcripts{ext}', f'{base}__utterance_excerpt_pairs{ext}'


def build_excerpts_from_transcript(input_csv_path, output_excerpt_csv_path=None, start_id=1):
	"""Process a single input transcript CSV.

	start_id: integer starting excerpt id to use for the first excerpt in
	this file. Returns (output_excerpt_csv_path, count, next_start_id).
	"""
	with open(input_csv_path, newline='', encoding='utf-8') as fh:
		reader = csv.DictReader(fh)
		rows = list(reader)

	current_question = None
	# Build excerpts: each excerpt starts with the interviewer question and
	# includes all utterances from then until the next question. Excerpt lines
	# are formatted as "Interviewer: ..." and "{participant}: ...".
	current_excerpt_lines = None
	current_excerpt_source = None
	current_excerpt_participants = set()
	source_val = None
	outputs = []

	# For combined transcripts: preserve original rows and add `excerpt_id`
	combined_rows = []
	current_excerpt_id = ''
	next_id = int(start_id)

	for row in rows:
		speaker = (row.get('speaker') or '').strip()
		utter = (row.get('utterance') or '').strip()
		src = (row.get('source') or '').strip()
		part = (row.get('participant') or '').strip()

		if not source_val and src:
			source_val = src

		# effective source: prefer the row's source, fall back to the first
		# seen non-empty source for the file
		effective_src = src or source_val or ''

		# interviewer prompt (utterance longer than four words): start a new excerpt (flush previous if present)
		if speaker.lower() == 'interviewer' and len(utter.split()) > 4:
			# flush previous excerpt
			if current_excerpt_lines is not None:
				combined = '\n'.join(x for x in current_excerpt_lines if x).strip()
				if combined:
					outputs.append({
						'excerpt_id': str(current_excerpt_id),
						'excerpt': combined,
						'source': current_excerpt_source or '',
						'participant': '|'.join(sorted(current_excerpt_participants)) if current_excerpt_participants else '',
						'question': current_question or '',
					})
			# start new excerpt with this question
			current_question = utter
			current_excerpt_lines = [f'Interviewer: {utter}']
			current_excerpt_source = effective_src
			current_excerpt_participants = set()
			# assign a new globally-unique id for this excerpt
			current_excerpt_id = str(next_id)
			next_id += 1

			# assign this row to the new excerpt
			out_row = dict(row)
			out_row['excerpt_id'] = current_excerpt_id
			combined_rows.append(out_row)

			continue

		# interviewer (non-question) within an excerpt: include if excerpt started
		if speaker.lower() == 'interviewer':
			if current_excerpt_lines is not None and utter:
				current_excerpt_lines.append(f'Interviewer: {utter}')
			out_row = dict(row)
			out_row['excerpt_id'] = current_excerpt_id or ''
			combined_rows.append(out_row)
			continue

		# participant utterance: include in current excerpt (if any)
		if speaker.lower() == 'participant':
			if not utter and not part:
				# still write row but with empty excerpt_id
				out_row = dict(row)
				out_row['excerpt_id'] = current_excerpt_id or ''
				combined_rows.append(out_row)
				continue
			if current_excerpt_lines is not None:
				participant_label = part or ''
				if utter:
					current_excerpt_lines.append(f'{participant_label}: {utter}')
				if part:
					current_excerpt_participants.add(part)
			out_row = dict(row)
			out_row['excerpt_id'] = current_excerpt_id or ''
			combined_rows.append(out_row)
			continue

		# other/unrecognized speaker: just write row with current excerpt id (if any)
		out_row = dict(row)
		out_row['excerpt_id'] = current_excerpt_id or ''
		combined_rows.append(out_row)

	# flush remaining excerpt at end of file (associate with last prompt if any)
	if current_excerpt_lines is not None:
		combined = '\n'.join(x for x in current_excerpt_lines if x).strip()
		if combined:
			outputs.append({
				'excerpt_id': str(current_excerpt_id),
				'excerpt': combined,
				'source': current_excerpt_source or '',
				'participant': '|'.join(sorted(current_excerpt_participants)) if current_excerpt_participants else '',
				'question': current_question or '',
			})

	# determine output path: place excerpt outputs in
	# data/excerpt_transcripts/<canonical_basename>__excerpt_transcripts.csv
	if output_excerpt_csv_path is None:
		input_basename = os.path.basename(input_csv_path)
		excerpt_basename, _ = canonical_output_basenames(input_basename)
		out_dir = os.path.join('data', 'excerpt_transcripts')
		os.makedirs(out_dir, exist_ok=True)
		output_excerpt_csv_path = os.path.join(out_dir, excerpt_basename)

	# write excerpt-pair CSV (with excerpt_id)
	fieldnames = ['excerpt_id', 'excerpt', 'source', 'participant', 'question']
	with open(output_excerpt_csv_path, 'w', newline='', encoding='utf-8') as outfh:
		writer = csv.DictWriter(outfh, fieldnames=fieldnames)
		writer.writeheader()
		for r in outputs:
			writer.writerow(r)

	# write combined transcripts with excerpt_id column
	combined_out_dir = os.path.join('data', 'excerpt_transcripts', 'utterance_excerpt_pairs')
	os.makedirs(combined_out_dir, exist_ok=True)
	_, mapping_basename = canonical_output_basenames(os.path.basename(output_excerpt_csv_path))
	combined_out_path = os.path.join(combined_out_dir, mapping_basename)
	combined_fieldnames = ['utterance_id', 'excerpt_id']
	with open(combined_out_path, 'w', newline='', encoding='utf-8') as cof:
		writer = csv.DictWriter(cof, fieldnames=combined_fieldnames)
		writer.writeheader()
		for r in combined_rows:
			# ensure all fields present
			out_r = {k: r.get(k, '') for k in combined_fieldnames}
			writer.writerow(out_r)

	return output_excerpt_csv_path, len(outputs), next_id


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-dir', default='data/combined_transcripts',
					 help='Directory containing input CSV files')
	args = parser.parse_args()

	input_dir = args.input_dir
	if not os.path.isdir(input_dir):
		print('Input directory does not exist:', input_dir)
		return

	# collect all .csv files in the directory (non-recursive)
	input_csv_files = sorted(
		os.path.join(input_dir, fn)
		for fn in os.listdir(input_dir)
		if fn.lower().endswith('.csv')
	)

	if not input_csv_files:
		print('No CSV files found in directory:', input_dir)
		return

	next_id = 1
	for input_csv_path in input_csv_files:
		output_csv_path = None
		try:
			output_csv_path, count, next_id = build_excerpts_from_transcript(input_csv_path, start_id=next_id)
			print(f'Wrote {count} pairs to: {output_csv_path}')
		except Exception as e:
			print(f'Error processing {input_csv_path}: {e}')


if __name__ == '__main__':
	main()

