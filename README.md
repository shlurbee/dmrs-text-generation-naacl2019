# dmrs-text-generation
Generating text from DMRS (minimal recursion semantics) representation.

Quick Start:

(1) Create and activate a virtual env, if desired
```
> virtualenv --python=python3 env
> source env/bin/activate
```

(2) Clone this repo, then from the root dir run `sh setup.sh` to install dependencies and get data.
```
> git clone git@github.com:shlurbee/dmrs-text-generation-naacl2019.git
> cd dmrs-text-generation
> sh setup.sh
```

(3) Next, the training data needs to be linearized, anonymized, and written to a parallel files
so they can be consumed by opennmt. (Run prep.sh to preprocess the default set of files, or look
at the commands for examples if you are testing different data.)
```
> sh scripts/prep.sh
```

(4) Train the model. You can use one of the `scripts/run_*` variants or write your own using
these as a guide. The model will be saved to the models/ directory after each epoch.
```
> sh scripts/run\_experiments.sh
```

(5) To evaluate, copy the name of the model you want to test and the file you want to test
it on into scripts/run\_eval.sh and run it. You can see the output in the results/ dir by
looking for files that start with the model name.
```
> sh scripts/run\_eval.sh
```

Currently (Sept 2018), the version of OpenNMT-py compatible with these scripts is:
```
commit 0ecec8b4c16fdec7d8ce2646a0ea47ab6535d308
```
https://github.com/OpenNMT/OpenNMT-py/commit/0ecec8b4c16fdec7d8ce2646a0ea47ab6535d308
