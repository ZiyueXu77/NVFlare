pkill -9 python
export NVFLARE_POC_WORKSPACE="/tmp/nvflare/fox_poc_10"
bash utils/log_memory.sh >>./fox_poc_10.txt &
python recipe_pt_hf_avg_stream.py
