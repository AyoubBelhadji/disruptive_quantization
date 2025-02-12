#!/bin/bash
cd ..
source activate disruptive_quantization
export LOCKFILE_PATH=/state/partition1/user/$USER/LOCKFILE
python main.py -rmd mnist$LLSUB_RANK
