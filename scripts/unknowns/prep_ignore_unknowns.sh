"""
Preprocessing script that anonymizes all unknown tokens. (Used in anon experiments.)
"""

OUTDIR=$1

# Make sure expected directories exist
mkdir -p $OUTDIR

# Linearize train, test data
python preprocessing.py data_penman/train.txt $OUTDIR/train
python preprocessing.py data_penman/dev.txt $OUTDIR/dev
echo 'done linearizing'

# Overwrite lines in training data that overlap with dev set
python scripts/remove_overlap.py \
    --train_files $OUTDIR/train-tgt.txt \
    --test_files $OUTDIR/dev-tgt.txt \
    --blacklist_file $OUTDIR/blacklist.txt

# Create file mapping anonymization tokens to most common surface form
python replacements.py \
    --infiles $OUTDIR/train-anon.txt \
    --outfile $OUTDIR/anon-replacements.json
