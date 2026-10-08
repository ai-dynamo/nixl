import functools
import os


@functools.lru_cache()
def find_nixl_root():
    """
    Find the NIXL installation root (`EP_NIXL_ROOT_DIR` or `NIXL_DIR`, checked for the NIXL headers), cached.
    """
    for env_name in ('EP_NIXL_ROOT_DIR', 'NIXL_DIR'):
        if env_name in os.environ:
            root = os.environ[env_name]
            for header in ('nixl.h', os.path.join('gpu', 'nixl_device.cuh')):
                assert os.path.isfile(os.path.join(root, 'include', header)), f'{env_name}={root} has no include/{header}'
            return root
    raise ModuleNotFoundError('Cannot find NIXL: set EP_NIXL_ROOT_DIR to the NIXL install prefix')


@functools.lru_cache()
def find_ucx_root():
    """
    Find the UCX installation root (`EP_UCX_ROOT_DIR`, `UCX_DIR`, `UCX_HOME`, else `ucx_info` on the PATH), cached.
    """
    def has_device_api(root: str) -> bool:
        return os.path.isfile(os.path.join(root, 'include', 'ucp', 'api', 'device', 'ucp_device_impl.h'))

    for env_name in ('EP_UCX_ROOT_DIR', 'UCX_DIR', 'UCX_HOME'):
        if env_name in os.environ:
            root = os.environ[env_name]
            assert has_device_api(root), f'{env_name}={root} has no UCX device API headers'
            return root
    import shutil
    ucx_info = shutil.which('ucx_info')
    if ucx_info is not None:
        root = os.path.dirname(os.path.dirname(os.path.realpath(ucx_info)))
        if has_device_api(root):
            return root
    raise ModuleNotFoundError('Cannot find UCX with the device API: set EP_UCX_ROOT_DIR to the UCX install prefix')


def find_nixl_lib_dir(nixl_root: str) -> str:
    """
    The directory holding `libnixl.so` inside a NIXL prefix (`lib` or a multiarch `lib/<triple>`).
    """
    import glob
    candidates = glob.glob(os.path.join(nixl_root, 'lib', 'libnixl.so')) + \
                 glob.glob(os.path.join(nixl_root, 'lib', '*-linux-gnu', 'libnixl.so')) + \
                 glob.glob(os.path.join(nixl_root, 'lib64', 'libnixl.so'))
    assert candidates, f'No libnixl.so under {nixl_root}'
    return os.path.dirname(candidates[0])
