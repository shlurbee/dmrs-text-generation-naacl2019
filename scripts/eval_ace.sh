#!/bin/bash

####
#  Helper script for postprocessing and running sacrebleu on ACE output
###

GOLD_FILENAME='data/dev/dev-orig.txt'

model_name="ace"
text_filename="results/$model_name.pred.text"
eval_filename="results/$model_name.sacrebleu"
echo "Evaluating $text_filename compared to $GOLD_FILENAME with SacreBLEU"
echo "cat $text_filename | sacrebleu $GOLD_FILENAME > $eval_filename"
cat $text_filename | sacrebleu $GOLD_FILENAME > $eval_filename
