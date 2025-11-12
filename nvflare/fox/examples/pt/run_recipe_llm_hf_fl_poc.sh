#!/bin/bash
# Script to run the HuggingFace LLM FL recipe with POC environment

# Kill any existing python processes
pkill -9 python

# Set POC workspace environment variable
export NVFLARE_POC_WORKSPACE="/tmp/nvflare/llm_hf_fl_poc"

# Optional: Log memory usage in background
# bash utils/log_memory.sh >>./llm_hf_fl_poc.txt &

# Run the recipe
python recipe_llm_hf_fl.py


