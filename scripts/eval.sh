#!/bin/bash

####
#  Helper script for postprocessing and running sacrebleu
###


if [ $# -ne 4 ]; then
  echo 'Usage: bash scripts/eval.sh {model_file_name} {eval_file_prefix} {anon_file} {gpu_id}'
  echo 'e.g., bash scripts/eval.sh models/v4_acc_71.91_ppl_10.01_e20.pt data/dev/dev data/anon-replacements.json 1'
  echo
  echo 'Expects parallel eval files: {prefix}-src.txt {prefix}-orig.txt {prefix}-anon.txt',
  echo '  which would have been created during preprocessing. (See scripts/prep.sh for example)'
  echo
  echo '{anon_file} is a mapping from anonymization tokens to surface forms that will replace '
  echo '  the tokens during de-anonymization. Create this file from your training data like so: '
  echo '> python replacements.py --infiles {space-delimited training data files} --outfile {anon_file}'
  exit
fi

# Get model filename prefix from command line (assumes model is stored in models/ dir)
MODEL_PATH=$1
EVAL_FILE_PREFIX=$2
ANON_REPLACEMENTS_MAP_FILENAME=$3
GPU=$4

# Extract base file name from model path (used for naming result files)
MODEL_NAME="${MODEL_PATH##*/}"

# Replace slashes with underscores in eval file name for use in output filename
EVAL_FILE_ID="${EVAL_FILE_PREFIX//\//_}"

# Gold and replacements filenames for all models
SOURCE_FILENAME="$EVAL_FILE_PREFIX-src.txt"
GOLD_FILENAME="$EVAL_FILE_PREFIX-orig.txt"
REPLACEMENTS_FILENAME="$EVAL_FILE_PREFIX-anon.txt"

echo "Source: $SOURCE_FILENAME"
echo "Gold: $GOLD_FILENAME"
echo "Replacements: $REPLACEMENTS_FILENAME"
echo "Replacements map: $ANON_REPLACEMENTS_MAP_FILENAME"

# Translator fails on blank lines, so replace them with "BLANK" before running eval. (requires -replace_unk)
BLANK='BLANK￨_'
SOURCE_FILENAME_SAFE="$SOURCE_FILENAME.safe"
cat $SOURCE_FILENAME | sed "s/^$/$BLANK/" > $SOURCE_FILENAME_SAFE
pred_filename="results/$MODEL_NAME.$EVAL_FILE_ID.pred.tokens"
text_filename_tmp="results/$MODEL_NAME.$EVAL_FILE_ID.pred.text.tmp"  # has BLANK for blank lines
text_filename="results/$MODEL_NAME.$EVAL_FILE_ID.pred.text"
eval_filename="results/$MODEL_NAME.$EVAL_FILE_ID.sacrebleu"
echo "Loading model from $MODEL_PATH"
echo "Using $MODEL_PATH to generate predictions, writing to $pred_filename..."
python OpenNMT-py/translate.py -model $MODEL_PATH -src $SOURCE_FILENAME_SAFE -output $pred_filename -replace_unk -gpu $GPU
echo "Postprocessing and writing text to $text_filename"
python postprocessing.py --infile $pred_filename --outfile $text_filename_tmp \
	--replacements $REPLACEMENTS_FILENAME --replacements_map $ANON_REPLACEMENTS_MAP_FILENAME
cat $text_filename_tmp | sed "s/^$BLANK$//" > $text_filename  # remove lines containing only "BLANK"
echo "Evaluating $text_filename compared to $GOLD_FILENAME with SacreBLEU"
echo "cat $text_filename | sacrebleu $GOLD_FILENAME > $eval_filename"
cat $text_filename | sacrebleu $GOLD_FILENAME > $eval_filename
