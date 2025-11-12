```commandline
nvflare provision -p project.yml
```

```commandline
cd /tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com/startup/
bash fl_admin.sh
```

```commandline
cd /tmp/nvflare/workspaces/edge_example/prod_00/
./start_all.sh
```

```commandline
python3 nvflare_implementation/jobs/pt_job_fedavg.py |& tee output.txt 
```
