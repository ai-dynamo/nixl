srun --mpi=pmix --gpus-per-node=8 --ntasks-per-node=8 -N 8 \
    --container-image="/mswg2/E2E/Regression_logs/squash/nixl_mpi.sqsh" \
    --container-mounts="/hpc:/hpc,/swgwork/eshukrun/nixl:/swgwork/eshukrun/nixl" \
    bash /swgwork/eshukrun/nixl/tools/perf/test/execute_8N.sh 2>&1 | tee log.txt
