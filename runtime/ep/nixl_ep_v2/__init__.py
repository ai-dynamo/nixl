import functools
import subprocess
import torch
import os

from .utils.find_pkgs import find_nixl_root, find_ucx_root

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Initialize
@functools.lru_cache()
def find_cuda_home() -> str:
    """
    Find the CUDA installation directory, cached.

    Returns:
        cuda_home: the CUDA installation path.
    """
    # TODO: reuse PyTorch API later
    # For some PyTorch versions, the original `_find_cuda_home` will initialize CUDA, which is incompatible with process forks
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # noinspection PyBroadException
        try:
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output(['which', 'nvcc'], stderr=devnull).decode().rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    assert cuda_home is not None
    return cuda_home


def init_jit():
    """
    Initialize the JIT compilation runtime. Sets up the CUDA, NIXL and UCX root paths for the JIT compiler.
    """
    # noinspection PyUnresolvedReferences
    import nixl_ep_v2._C as _C
    library_root_path = os.path.dirname(os.path.abspath(__file__))
    _C.init_jit(library_root_path,  # Library root directory path
                find_cuda_home(),   # CUDA home
                find_nixl_root(),   # NIXL root
                find_ucx_root())    # UCX root

# Run initialization
init_jit()


# Import APIs after initialization
from .buffers.elastic import ElasticBuffer, EPHandle
# noinspection PyUnresolvedReferences
from .utils.event import EventOverlap, EventHandle
from .utils.envs import get_physical_domain_size, get_logical_domain_size

# noinspection PyUnresolvedReferences
from nixl_ep_v2._C import topk_idx_t

__version__ = '2.1.0'
