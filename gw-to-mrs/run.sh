#!/bin/bash

###
# Commands used to generate DMRS graphs for silver training data.
#
# Note: Commands were pasted into this file as they were run.
# The script hasn't been tested straight through.
###

# Print commands when executing them
set -o xtrace

# Extract text from files and split into sentences
python gw_to_sentences.py

# Get a random subset of 1M
cat sentences.txt | shuf | head -n 1000000 > sentences1M.txt

# Check vocab size
cat sentences1M.txt | awk -F'\t' '{print $2}' | tr -d '[:punct:]' | tr -d '0123456789' | tr '[:upper:]' '[:lower:]' | tr ' ' '\n' | sort | uniq | wc -l

# Check number of words
cat sentences1M.txt | awk -F'\t' '{print $2}' | tr ' ' '\n' | wc -l

# Shuffle sentences and write to separate file
cat sentences1M.txt | shuf | awk -F'\t' '{print $2}' > sentences1M_text.txt

# Split sentence file into shards that can be processed in parallel
split -l 50000 sentences1M_text.txt data/sentences.

# Parse some of the splits
declare -a arr=("aa" "ab" "ac" "ad" "ae")

for i in "${arr[@]}"
do
    FILENAME="data/sentences.$i"
    python3 ../mrs-to-penman/mrs_to_penman.py \
            --ace-binary ../mrs-to-penman/ace-0.9.25/ace \
            -g /home/vhajdik/dmrs-text-generation/mrs-to-penman/erg-1214-x86-64-0.9.25.dat \
            -i $FILENAME \
            -p ../mrs-to-penman/parameters.json \
            > $FILENAME.mrs &
done

# Concatenate all parse files into one huge file to get ready for linearization
cat data/sentences.*.mrs > parsed.txt
