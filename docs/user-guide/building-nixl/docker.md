---
title: Docker
description: Build and run NIXL in a Docker container.
---

Clone the repository, then build the Docker container:

```bash
./contrib/build-container.sh
```

By default, the container is built with Ubuntu 24.04. To build for Ubuntu 22.04:

```bash
./contrib/build-container.sh --os ubuntu22
```

To see all available options:

```bash
./contrib/build-container.sh -h
```

<Tip>
The container includes a prebuilt Python wheel in `/workspace/dist` for installing the Python bindings.
</Tip>

<Note>
For backend-specific build instructions, see [NIXL Backends](/docs/user-guide/backend-selection).
</Note>
