###
# Preprocessing script that anonymizes unknown tokens that occur less than two times
# in the training data. Other tokens are inserted in the training data as-is.
###

OUTDIR=$1

# Make sure expected directories exist
mkdir -p $OUTDIR

# Linearize train, dev, test data
python preprocessing.py data_penman/train.txt $OUTDIR/train
python preprocessing.py data_penman/dev.txt $OUTDIR/dev
echo 'done linearizing'

# Build vocab from training data
python replace_rare.py vocab --vocabfile $OUTDIR/vocab.txt --infile $OUTDIR/train-tgt.txt

# Use vocab to replace rare tokens in train/dev/test sets, anonymizing only those unknown tokens with freq 1
python replace_rare.py replace --vocabfile $OUTDIR/vocab.txt --infile $OUTDIR/train --min_freq 2
python replace_rare.py replace --vocabfile $OUTDIR/vocab.txt --infile $OUTDIR/dev --min_freq 2

# Overwrite lines in training data that overlap with dev set
python scripts/remove_overlap.py \
    --train_files $OUTDIR/train-tgt.txt \
    --test_files $OUTDIR/dev-tgt.txt \
    --blacklist_file $OUTDIR/blacklist.txt

# Create file mapping anonymization tokens to most common surface form
python replacements.py \
    --infiles $OUTDIR/train-anon.txt \
    --outfile $OUTDIR/anon-replacements.json
