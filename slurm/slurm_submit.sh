#!/bin/bash
cd ..
source activate idp
python main.py -mnd mnist_gen$LLSUB_RANK
