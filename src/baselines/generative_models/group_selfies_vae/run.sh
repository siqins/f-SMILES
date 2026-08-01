#!/bin/bash

# add sys.path
export PYTHONPATH="$(pwd)"

conda activate py311pt21

#nohup python -u $run_file --run smcv > "smcv.txt" 2>&1 &
#nohup python -u "scripts/step_00_random_seed.py" --data_path "data/organize_data" --save_path "data/organize_data" --result_save_path "experiments" --batch_size 64 --epoch 200 > "00_random_seed.txt" 2>&1 &
nohup python -u "tune.py" > "tune.txt" 2>&1 &
