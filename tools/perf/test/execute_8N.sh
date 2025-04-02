
unset UCX_TLS

export NCCL_IB_QPS_PER_CONNECTION=2 \
NCCL_IB_LH_QPS_PER_CONNECTION=8 \
NCCL_IB_SPLIT_DATA_ON_QPS=0 \
NCCL_IB_ADAPTIVE_ROUTING=1 \
NCCL_IB_LH_LATENCY=5 \
NCCL_IB_TC=96 \
NCCL_IB_LH_TC=144 \
NCCL_IB_GID_INDEX=3  \
NCCL_SOCKET_IFNAME="eno8303" \
OMPI_MCA_btl_tcp_if_include=eno8303 \
NCCL_IB_HCA="mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MELLANOX_VISIBLE_DEVICES=0,3,4,5,6,9,10,11 \
UCX_RNDV_SCHEME=get_zcopy \
UCX_IB_GID_INDEX=3 \
UCX_TLS=^gdr_copy \
UCX_LOG_LEVEL=error 

#echo "Testing custom traffic perftest"
#python /swgwork/eshukrun/nixl/tools/perf/cli.py ct-perftest /swgwork/eshukrun/nixl/tools/perf/test/test_8N.yaml --verify-buffers 

echo "Testing multi-custom traffic perftest"
python /swgwork/eshukrun/nixl/tools/perf/cli.py sequential-ct-perftest /swgwork/eshukrun/nixl/tools/perf/matrices/metadata.yaml --verify-buffers 

#python /swgwork/eshukrun/nixl/test/python/nixl_api_test.py

#python /swgwork/eshukrun/nixl_perftest/python/alltoallv_perftest.py

#python /swgwork/eshukrun/nixl_perftest/python/playground.py