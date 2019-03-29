#!/bin/bash

# You probabaly want to create and activate a virtual environment first:
# virtualenv --python=python3 ./env
# source env/bin/activate

# install python dependencies
pip install -r requirements.txt
python -c 'import nltk; nltk.download("perluniprops"); nltk.download("nonbreaking_prefixes")'

# Fetch scripts that convert tsdb data to Penman graphs
git clone git@github.com:goodmami/mrs-to-penman.git

# Note: 1214 branch instead of trunk since trunk is incomplete
svn co http://svn.delph-in.net/erg/tags/1214/tsdb/gold mrs-to-penman/profiles

# generate Penman-serialized representations
cp config/convert_redwoods_params.json mrs-to-penman/parameters.json
cd mrs-to-penman
./convert-redwoods.sh
rm -rf ../data_penman
mv out ../data_penman
cd ..

# concatenate files to get one file per data split
cat data_penman/train/*.txt > data_penman/train.txt
cat data_penman/dev/*.txt > data_penman/dev.txt
cat data_penman/test/*.txt > data_penman/test.txt

# install opennmt
git clone https://github.com/OpenNMT/OpenNMT-py
cd OpenNMT-py
git checkout 0ecec8b4c16fdec7d8ce2646a0ea47ab6535d308
pip install -r requirements.txt
cd ..

# fix to prevent error when this version of opennmt encounters missing feat
sed -i '69s/.*/        example_values = ([ex.get(k, []) for k in keys] for ex in examples_iter)/' OpenNMT-py/onmt/io/TextDataset.py

# preprocess sample file to test setup
python preprocessing.py data/sample/sample.txt data/sample/sample

# download mrs parsed data
if [ ! -x /usr/bin/wget ] ; then
    command -v wget >/dev/null 2>&1 || { echo >&2 "ERROR: Could not download parsed.txt because wget not found"; exit 1; }
fi
mkdir data_gw
wget -O data_gw/parsed.txt https://s3-us-west-1.amazonaws.com/dmrs-text-generation/parsed.txt

echo "Setup complete."
echo
echo "When you're ready to preprocess the data splits:"
echo "sh scripts/prep.sh"

echo
