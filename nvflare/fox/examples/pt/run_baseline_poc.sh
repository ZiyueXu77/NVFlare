pkill -9 python
bash utils/log_memory.sh >>./baseline_poc.txt &
python3 recipe_llm_hf_fl.py
