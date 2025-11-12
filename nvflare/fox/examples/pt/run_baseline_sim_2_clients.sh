pkill -9 python
bash utils/log_memory.sh >>./baseline.txt &
python3 llm_hf_fl_job.py --client_ids 1 2
