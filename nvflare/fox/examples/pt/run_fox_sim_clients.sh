pkill -9 python
bash utils/log_memory.sh >>./fox_sim.txt &
python pt_hf_avg_stream.py
