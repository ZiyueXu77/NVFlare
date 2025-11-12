#!/bin/bash
# Script to run the HF FL recipe with POC environment

# Kill any existing python processes
pkill -9 python

# Set POC workspace environment variable
export NVFLARE_POC_WORKSPACE="/tmp/nvflare/fox_hf_fl_poc"

# Optional: Log memory usage in background
# bash utils/log_memory.sh >>./hf_fl_poc.txt &

# Run the recipe
python recipe_hf_fl.py


