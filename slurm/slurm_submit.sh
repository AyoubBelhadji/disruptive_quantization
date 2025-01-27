#!/bin/bash
cd ..
source activate idp
export LOCKFILE_PATH=/state/partition1/user/$USER/LOCKFILE
python main.py -rmd test
