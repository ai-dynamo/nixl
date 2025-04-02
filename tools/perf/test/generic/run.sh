export config="/swgwork/eshukrun/nixl/tools/perf/matrices_folders/llama-3b-8k-to-64k-16ptp4-32dtp8/metadata.yaml"
NNODES=6

srun --mpi=pmix --gpus-per-node=8 --ntasks-per-node=8 -N $NNODES \
    --container-image="/mswg2/E2E/Regression_logs/squash/nixl_mpi.sqsh" \
    --container-mounts="/hpc:/hpc,/swgwork/eshukrun/nixl:/swgwork/eshukrun/nixl" \
    bash /swgwork/eshukrun/nixl/tools/perf/test/generic/execute.sh 2>&1 | grep -v "The hostname of the client socket cannot be retrieved" | grep -v "The client socket has failed" | grep -v "Using backend" | tee /swgwork/eshukrun/nixl/tools/perf/test/generic/log.txt
