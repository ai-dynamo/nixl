# Slurm Test Scripts

Scripts for running NIXL tests on a Slurm cluster using Docker containers.

## Quick Start

```bash
# Run tests on Slurm cluster
./slurm_test_orchestrator.sh ".gitlab/test_cpp.sh /opt/nixl" -p rock -i "harbor.mellanox.com/ucx/x86_64/pytorch:25.02-py3"
```

## Scripts

### slurm_test_orchestrator.sh
Main entry point that:
- Copies Slurm configuration from head node
- Launches Slurm client container
- Monitors job execution

**Usage:**
```bash
./slurm_test_orchestrator.sh <test_cmd> [-p <partition>] [-i <docker_image>] [-t <timeout>]
```

### slurm_client_runner.sh
Runs inside the Slurm client container to:
- Submit job via `sbatch`
- Monitor job status
- Return results

## Architecture

```
Jenkins Agent
    ├── slurm_test_orchestrator.sh
    │   └── Docker: Slurm client container
    │       └── slurm_client_runner.sh
    │           └── sbatch → Compute node → Test container
```

## Environment Variables

**CI Variables:**
- `NIXL_INSTALL_DIR`: NIXL installation directory (default: `/opt/nixl`)
- `UCX_INSTALL_DIR`: UCX installation directory (default: `/usr/local`)
- `SLURM_PARTITION`: Slurm partition (default: `rock`)
- `SLURM_TIMEOUT`: Job timeout (default: `01:00:00`)
- `DEBUG`: Enable debug output (0-9)

**Git Reference** (checked in order):
- `sha1`, `GIT_COMMIT`, `GIT_BRANCH`, `CHANGE_BRANCH` (default: `main`)

## Custom Docker Image

The orchestrator uses a custom Slurm client image with pre-configured user and permissions:

```bash
# Build and push custom image
cd contrib/slurm
docker build -t harbor.mellanox.com/ucx/x86_64/slurm-client:v1.0 .
docker push harbor.mellanox.com/ucx/x86_64/slurm-client:v1.0
```

**Benefits:**
- Pre-configured `svc-nixl` user
- Fixed MUNGE permissions
- Faster startup
- Cleaner logs 