import argparse

def add_common_args(subparser: argparse.ArgumentParser):
    subparser.add_argument("--model", type=str, help="Model name (e.g., 'llama3.1-8b')")
    subparser.add_argument("--model_config", type=str, help="Path to the model config YAML file")

def add_nixl_bench_args(subparser: argparse.ArgumentParser):
    subparser.add_argument("--backend", type=str, help="Communication backend [UCX, UCX_MO] (default: UCX)")
    subparser.add_argument("--worker_type", type=str, help="Worker to use to transfer data [nixl, nvshmem] (default: nixl)")
    subparser.add_argument("--initiator_seg_type", type=str, help="Memory segment type for initiator [DRAM, VRAM] (default: DRAM)")
    subparser.add_argument("--target_seg_type", type=str, help="Memory segment type for target [DRAM, VRAM] (default: DRAM)")
    subparser.add_argument("--scheme", type=str, help="Communication scheme [pairwise, manytoone, onetomany, tp] (default: pairwise)")
    subparser.add_argument("--mode", type=str, help="Process mode [SG (Single GPU per proc), MG (Multi GPU per proc)] (default: SG)")
    subparser.add_argument("--op_type", type=str, help="Operation type [READ, WRITE] (default: WRITE)")
    subparser.add_argument("--check_consistency", action="store_true", help="Enable consistency checking")
    subparser.add_argument("--total_buffer_size", type=int, help="Total buffer size (default: 8GiB)")
    subparser.add_argument("--start_block_size", type=int, help="Starting block size (default: 4KiB)")
    subparser.add_argument("--max_block_size", type=int, help="Maximum block size (default: 64MiB)")
    subparser.add_argument("--start_batch_size", type=int, help="Starting batch size (default: 1)")
    subparser.add_argument("--max_batch_size", type=int, help="Maximum batch size (default: 1)")
    subparser.add_argument("--num_iter", type=int, help="Number of iterations (default: 1000)")
    subparser.add_argument("--warmup_iter", type=int, help="Number of warmup iterations (default: 100)")
    subparser.add_argument("--num_threads", type=int, help="Number of threads used by benchmark (default: 1)")
    subparser.add_argument("--num_initiator_dev", type=int, help="Number of devices in initiator processes (default: 1)")
    subparser.add_argument("--num_target_dev", type=int, help="Number of devices in target processes (default: 1)")
    subparser.add_argument("--enable_pt", action="store_true", help="Enable progress thread")
    subparser.add_argument("--device_list", type=str, help="Comma-separated device names (default: all)")
    subparser.add_argument("--runtime_type", type=str, help="Type of runtime to use [ETCD] (default: ETCD)")
    subparser.add_argument("--etcd-endpoints", type=str, help="ETCD server URL for coordination (default: http://localhost:2379)")
    

