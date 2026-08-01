#!/bin/bash

# add sys.path
export PYTHONPATH="$(pwd)"

conda activate py311pt21

nohup python -u "02_xtb_run.py" > "02_xtb_run.txt" 2>&1 &
