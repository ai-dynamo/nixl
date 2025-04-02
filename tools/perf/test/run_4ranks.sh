srun --mpi=pmix --gpus-per-node=8 --ntasks-per-node=4 -N 1 \
    --container-image="/mswg2/E2E/Regression_logs/squash/nixl_mpi.sqsh" \
    --container-mounts="/hpc:/hpc,/swgwork/eshukrun/nixl:/swgwork/eshukrun/nixl" \
    bash /swgwork/eshukrun/nixl/tools/perf/test/execute_4ranks.sh 2>&1 | tee log.txt
