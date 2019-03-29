###
# Preprocess training and evaluation data.
###

# Make sure expected directories exist
mkdir -p data/train
mkdir -p data/dev
mkdir -p data/test
mkdir -p data_gw/train
mkdir -p data_wsj/dev/
mkdir -p data_wsj/test/

# Create dev set of only wsj examples (since these most closely match the silver domain)
cat data_penman/dev/wsj*.txt > data_penman/dev_wsj.txt
cat data_penman/test/wsj*.txt > data_penman/test_wsj.txt

# Linearize train, dev, test data
# "--with_blanks" inserts blank lines when the input can't be parsed
# to make sure line numbers don't change (useful for comparison with parser)
python preprocessing.py data_penman/train.txt data/train/train
python preprocessing.py --with_blanks data_penman/dev.txt data/dev/dev
python preprocessing.py --with_blanks data_penman/test.txt data/test/test
python preprocessing.py --with_blanks data_penman/dev_wsj.txt data_wsj/dev/dev
python preprocessing.py --with_blanks data_penman/test_wsj.txt data_wsj/test/test
python preprocessing.py data_gw/parsed.txt data_gw/train/train
echo 'done linearizing'

# Overwrite lines in gold and silver training data that overlap with dev set,
# to avoid leaking answers to the model. The script first creates a blacklist
# of sentences that appear in both, then rewrites the training data, inserting
# blank lines in place of those found in the blacklist.
python scripts/remove_overlap.py \
    --train_files data/train/train-tgt.txt \
    --test_files data/dev/dev-tgt.txt data/test/test-tgt.txt \
    --blacklist_file data/blacklist.txt
python scripts/remove_overlap.py \
    --train_files data_gw/train/train-tgt.txt \
    --test_files data/dev/dev-tgt.txt data/test/test-tgt.txt \
    --blacklist_file data/blacklist_gw.txt

# Create file mapping anonymization tokens to most common surface form, computed
# across one copy of gold + one copy of silver data. The --outfile value here will
# be passed as an argument to the eval.sh script and is used during de-anonymization.
# Note: if you use different training data you probably want to regenerate this file
# to match whatever training data you're using.
python replacements.py \
	--infiles data/train/train-anon.txt data_gw/train/train-anon.txt \
	--outfile data/anon-replacements.json

# New opennmt files cannot be created unless old ones are removed first
rm data/opennmt.*
rm data_gw/opennmt.*
