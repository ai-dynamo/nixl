#!/bin/bash
# ISR1-PRE: 8 GPUs with 8 dedicated mlx5 NICs (1:1 mapping)
# Based on your old exec-cetera script
index=$SLURM_LOCALID
export CUDA_VISIBLE_DEVICES=$index

if (( $index == 0 )); then
    export UCX_NET_DEVICES=mlx5_0:1
elif (( $index == 1 )); then
    export UCX_NET_DEVICES=mlx5_3:1
elif (( $index == 2 )); then
    export UCX_NET_DEVICES=mlx5_4:1
elif (( $index == 3 )); then
    export UCX_NET_DEVICES=mlx5_5:1
elif (( $index == 4 )); then
    export UCX_NET_DEVICES=mlx5_6:1
elif (( $index == 5 )); then
    export UCX_NET_DEVICES=mlx5_9:1
elif (( $index == 6 )); then
    export UCX_NET_DEVICES=mlx5_10:1
elif (( $index == 7 )); then
    export UCX_NET_DEVICES=mlx5_11:1
fi

echo "Rank $SLURM_PROCID: GPU=$index NIC=$UCX_NET_DEVICES"
exec "$@"
