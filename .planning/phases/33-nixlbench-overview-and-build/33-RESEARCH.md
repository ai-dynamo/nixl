# Phase 33: NIXLBench Overview and Build - Research

**Researched:** 2026-04-07
**Status:** Complete

## Research Questions

### Q1: What is NIXLBench and what are its key features?

NIXLBench is a comprehensive benchmarking tool for NIXL that uses etcd for coordination. It provides performance testing across multiple communication backends and storage systems for evaluating high-performance data transfer scenarios in distributed computing environments.

**Key features (grouped per D-08):**
- **Backends:** Network (UCX, GPUNETIO, Mooncake, Libfabric) and Storage (GDS, GDS_MT, POSIX, HF3FS, OBJ, GUSLI, Azure Blob)
- **Communication patterns:** Pairwise, many-to-one, one-to-many, TP (tensor parallel)
- **Memory types:** CPU (DRAM) and GPU (VRAM) transfers
- **Worker types:** NIXL worker (full-featured) and NVSHMEM worker (GPU-focused, VRAM-only)
- **Coordination:** etcd-based worker coordination for containerized/cloud-native environments
- **Performance:** Multi-threading, VMM memory allocation, latency percentiles, data consistency validation

### Q2: What are the system requirements?

**Hardware:**
- CPU: x86_64 or aarch64
- Memory: 8GB RAM minimum (16GB+ for compilation)
- Storage: 20GB free disk space
- GPU: NVIDIA GPU with CUDA support (for GPU features)
- Network: InfiniBand/Ethernet (UCX/GPUNetIO/Mooncake), EFA in AWS (UCX/Libfabric)

**Software:**
- OS: Ubuntu 22.04/24.04 LTS or RHEL-based
- Docker: 20.10+ (for container builds)
- CUDA Toolkit: 12.8+
- Python: 3.12+
- Git

### Q3: What are the Docker build options?

From `benchmark/nixlbench/contrib/build.sh`, the available options are:

| Option | Description | Default |
|--------|-------------|---------|
| `--nixl <path>` | Path to NIXL source directory | Parent NIXL directory |
| `--nixlbench <path>` | Path to NIXLBench source directory | Current directory |
| `--ucx <path>` | Path to custom UCX source | Uses base image UCX |
| `--build-type <type>` | `debug` or `release` | `release` |
| `--base-image <image>` | Base Docker image | `nvcr.io/nvidia/cuda-dl-base` |
| `--base-image-tag <tag>` | Base image tag | `25.10-cuda13.0-devel-ubuntu24.04` |
| `--arch <arch>` | `x86_64` or `aarch64` | Auto-detected |
| `--python-versions <ver>` | Comma-separated Python versions | `3.12` |
| `--tag <tag>` | Custom Docker image tag | Auto-generated |
| `--no-cache` | Disable Docker build cache | Cache enabled |

**Per D-04:** Show essentials only on the page (basic invocation + `--build-type`, `--arch`), link to README for full table.

### Q4: What does the native build process look like?

Native build requires:
1. Build NIXL first (link to existing docs per D-05)
2. Build NIXLBench with meson:
   ```
   meson setup build -Dnixl_path=/usr/local/nixl --buildtype=release
   cd build && ninja && sudo ninja install
   ```

**Meson build options:**
- `nixl_path`: Path to NIXL installation
- `cudapath_inc`, `cudapath_lib`, `cudapath_stub`: CUDA paths
- `etcd_inc_path`, `etcd_lib_path`: etcd C++ client paths
- `nvshmem_inc_path`, `nvshmem_lib_path`: NVSHMEM paths
- `buildtype`: debug/release/debugoptimized
- `prefix`: Installation prefix

**Core dependencies:** NIXL, UCX, CUDA (>=12.8), CMake (>=3.20), Meson, Ninja, etcd-cpp-api, GFlags, OpenMP.

**Optional dependencies:** LibFabric, DOCA, AWS SDK C++, Azure SDK for C++, GDS, GUSLI, NVSHMEM, hwloc.

### Q5: What Fern MDX patterns must be followed?

From existing docs analysis:
- **Frontmatter:** YAML with `title:` and `description:` (no other fields)
- **Components:** `<Tabs>` for multi-path instructions, `<Note>`, `<Tip>`, `<Warning>` for callouts
- **Links:** Relative paths like `./building-nixl/docker` or absolute `/docs/user-guide/backend-selection`
- **No GitHub-specific constructs:** No `<!-- -->` HTML comments, no `<details>` collapsibles, no bare anchor links
- **Code blocks:** Triple backtick with language identifier

### Q6: What cross-links are needed?

**Backend links (first mention per page) — paths from `docs/index.yml`:**
- UCX: `/docs/user-guide/backends/ucx`
- Libfabric: `/docs/user-guide/backends/libfabric`
- Mooncake: `/docs/user-guide/backends/mooncake`
- DOCA GPUNetIO: `/docs/user-guide/backends/gpunetio`
- GPUDirect Storage (GDS): `/docs/user-guide/backends/gds`
- GPUDirect Storage MT: `/docs/user-guide/backends/gds-mt`
- POSIX: `/docs/user-guide/backends/posix`
- HF3FS: `/docs/user-guide/backends/hf3fs`
- OBJ: `/docs/user-guide/backends/obj`
- Azure Blob: `/docs/user-guide/backends/azure-blob`
- GUSLI: `/docs/user-guide/backends/gusli`

**Other cross-links:**
- Building NIXL from Source: `/docs/user-guide/building-nixl`
- Metadata Exchange with ETCD: `/docs/user-guide/etcd-metadata-exchange`

### Q7: What nav changes are needed?

Current `docs/index.yml` under NIXLBench:
```yaml
- section: NIXLBench
  collapsed: open-by-default
  path: development/benchmarking/nixlbench/index.md
  contents:
    - page: Usage and Troubleshooting
      path: development/benchmarking/nixlbench/usage.md
```

Need to add `build.md` entry per D-01:
```yaml
- section: NIXLBench
  collapsed: open-by-default
  path: development/benchmarking/nixlbench/index.md
  contents:
    - page: Building NIXLBench
      path: development/benchmarking/nixlbench/build.md
    - page: Usage and Troubleshooting
      path: development/benchmarking/nixlbench/usage.md
```

### Q8: What existing page patterns should be replicated?

The `docs/user-guide/building-nixl/docker.md` page is very concise (30 lines): frontmatter, one paragraph, code block, a `<Tip>` and `<Note>`. The section index page (`building-nixl/index.md`) is also concise: frontmatter, bullet list of sub-pages, then a build options table.

The overview page (`index.md`) should follow the section-index pattern but with more narrative content (problem-first per D-02) since it's describing a tool, not just listing sub-pages.

The build page (`build.md`) should follow Docker page conciseness but use `<Tabs>` for Docker vs Native side-by-side per D-03.

## Validation Architecture

### Dimension 1: Structural Correctness
- Both files exist at correct paths
- YAML frontmatter with `title:` and `description:`

### Dimension 2: Content Accuracy
- NIXLBench features match README source
- Build options match `build.sh` actual flags
- System requirements match README

### Dimension 3: Cross-Reference Integrity
- Backend names link to correct User Guide pages
- NIXL build link works
- etcd link works

### Dimension 4: Fern Compatibility
- No GitHub-only constructs
- `<Tabs>` component used correctly
- `fern check` passes

### Dimension 5: Non-Duplication
- Build page links to existing NIXL build docs, not repeating them
- Docker section shows essentials, links to README for full table

## Key Findings

1. **Nav update needed:** `docs/index.yml` must add `build.md` under NIXLBench section
2. **New file needed:** `docs/development/benchmarking/nixlbench/build.md` does not exist yet
3. **Stub replacement:** `docs/development/benchmarking/nixlbench/index.md` has stub content to be replaced
4. **D-06 conflict with roadmap SC-1:** ROADMAP success criterion 1 says overview page should describe system requirements, but CONTEXT.md D-06 says system requirements go on the build page. CONTEXT.md decisions (user preferences) take precedence — system requirements go on build page.
5. **Terminology:** Use `etcd` (lowercase) in prose, `plug-in` (hyphenated), consistent backend capitalization per QS-01

## RESEARCH COMPLETE
