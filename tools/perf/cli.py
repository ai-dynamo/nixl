import click
import yaml
import logging
from pathlib import Path
from dist_utils import dist_utils
from custom_traffic_perftest import CTPerftest, TrafficPattern
from multi_custom_traffic_perftest import MultiCTPerftest

@click.group()
@click.option('--debug/--no-debug', default=False, help="Enable debug logging")
def cli(debug):
    """NIXL Performance Testing CLI"""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set level for all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--verify-buffers/--no-verify-buffers', default=False, help="Verify buffer contents after transfer")
@click.option('--print-recv-buffers/--no-print-recv-buffers', default=False, help="Print received buffer contents")
def ct_perftest(config_file, verify_buffers, print_recv_buffers):
    """Run custom traffic performance test using patterns defined in YAML config"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    
    tp_config = config.get('traffic_pattern')
    if tp_config is None:
        raise ValueError("Config file must contain 'traffic_pattern' key")
    
    required_fields = ['matrix_file', 'shards', 'mem_type', 'xfer_op']
    missing_fields = [field for field in required_fields if field not in tp_config]
        
    if missing_fields:
        raise ValueError(f"Traffic pattern missing required fields: {missing_fields}")
    
    iters = config.get('iters', 1)
    warmup_iters = config.get('warmup_iters', 0)
    pattern = TrafficPattern(
        matrix_file=Path(tp_config['matrix_file']),
        shards=tp_config['shards'],
        mem_type=tp_config.get('mem_type', 'dram').lower(),
        xfer_op=tp_config.get('xfer_op', 'WRITE').upper(),
    )
    
    perftest = CTPerftest(pattern, iters=iters, warmup_iters=warmup_iters)
    perftest.run(verify_buffers=verify_buffers, print_recv_buffers=print_recv_buffers)
    dist_utils.destroy_dist()

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--verify-buffers/--no-verify-buffers', default=False, help="Verify buffer contents after transfer")
@click.option('--print-recv-buffers/--no-print-recv-buffers', default=False, help="Print received buffer contents")
def multi_ct_perftest(config_file, verify_buffers, print_recv_buffers):
    """Run custom traffic performance test using patterns defined in YAML config"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'traffic_patterns' not in config:
        raise ValueError("Config file must contain 'traffic_patterns' key")
    
    patterns = []
    for instruction_config in config['traffic_patterns']:
        tp_config = instruction_config
        required_fields = ['matrix_file', 'shards', 'mem_type', 'xfer_op']
        missing_fields = [field for field in required_fields if field not in tp_config]
        
        if missing_fields:
            raise ValueError(f"Traffic pattern missing required fields: {missing_fields}")
        
        pattern = TrafficPattern(
            matrix_file=Path(tp_config['matrix_file']),
            shards=tp_config['shards'],
            mem_type=tp_config.get('mem_type', 'dram').lower(),
            xfer_op=tp_config.get('xfer_op', 'WRITE').upper(),
            sleep_after_launch_sec=tp_config.get('sleep_after_launch_sec', 0),
        )
        patterns.append(pattern)
    
    perftest = MultiCTPerftest(patterns)
    perftest.run(verify_buffers=verify_buffers, print_recv_buffers=print_recv_buffers)
    dist_utils.destroy_dist()

if __name__ == '__main__':
    cli()
