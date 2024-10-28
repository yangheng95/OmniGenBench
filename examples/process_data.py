# -*- coding: utf-8 -*-
# file: process_data.py
# time: 18:02 04/10/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import json

def process_and_merge_sequences(input_file_path, output_file_path, chunk_size=1000):
    """
    Reads the input file line by line, processes and merges the 5utr, CDS, and 3utr sequences,
    and writes the merged sequence to the output file in chunks.

    :param input_file_path: Path to the input file containing the sequences.
    :param output_file_path: Path to the output file to write the merged sequences.
    :param chunk_size: Number of lines to process in one chunk. Default is 1000.
    """

    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        lines_to_process = []

        for line in infile:
            # Collect lines until we reach the chunk size
            lines_to_process.append(line.strip())

            if len(lines_to_process) >= chunk_size:
                # Process the current chunk
                process_chunk(lines_to_process, outfile)
                lines_to_process = []

        # Process any remaining lines after the loop
        if lines_to_process:
            process_chunk(lines_to_process, outfile)


def process_chunk(lines, outfile):
    """
    Process each line in the chunk by merging the 5utr, CDS, and 3utr sequences,
    and write the result to the output file.

    :param lines: A list of lines to process.
    :param outfile: The output file to write the merged sequences.
    """
    for line in lines:
        # Assuming each line is a valid JSON string like in the original example
        sequence_data = json.loads(line)

        # Merge the sequences (5'UTR, CDS, 3'UTR)
        merged_sequence = sequence_data["5utr_seq"] + sequence_data["CDS_seq"] + sequence_data["3utr_seq"]

        # Write the merged sequence to the output file
        outfile.write(merged_sequence + "\n")


# Example usage
input_file = "animal_seq.json.txt"  # Path to your large input file
output_file = "all.fa.animal.txt"  # Output file where merged sequences will be saved
process_and_merge_sequences(input_file, output_file, chunk_size=10000)
