/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 */
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "spdk/env.h"
#include "spdk/memory.h"
#include "../../../../src/plugins/spdk/spdk_env/shared_memory.h"

#define CHECK(c)                                                 \
    do {                                                         \
        if (!(c)) {                                              \
            fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); \
            return 1;                                            \
        }                                                        \
    } while (0)

static int registrations, unregistrations;

static int
notify(void *, spdk_mem_map *, spdk_mem_map_notify_action action, void *addr, size_t len) {
    uint64_t offset;
    if (spdk_mem_get_fd_and_offset(addr, &offset) < 0 ||
        spdk_mem_get_fd_and_offset(static_cast<char *>(addr) + len - 1, &offset) < 0) {
        return -EINVAL;
    }
    if (action == SPDK_MEM_MAP_NOTIFY_REGISTER) {
        ++registrations;
    } else {
        ++unregistrations;
    }
    return 0;
}

int
main() {
    spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    CHECK(spdk_env_init(&opts) == 0);
    spdk_mem_map_ops ops = {};
    ops.notify_cb = notify;
    auto *map = spdk_mem_map_alloc(0, &ops, nullptr);
    CHECK(map != nullptr);
    auto *buf = static_cast<char *>(spdk_dma_zmalloc(8192, 4096, nullptr));
    CHECK(buf != nullptr);
    CHECK(registrations > 0);
    uint64_t offset = 0;
    int fd = spdk_mem_get_fd_and_offset(buf, &offset);
    CHECK(fd >= 0 && (fcntl(fd, F_GETFD) & FD_CLOEXEC));
    CHECK(offset % 4096 == 0);
    void *view = mmap(nullptr, 8192, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
    CHECK(view != MAP_FAILED);
    memset(buf, 0x5a, 8192);
    CHECK(memcmp(buf, view, 8192) == 0);
    static_cast<char *>(view)[8191] = 0x33;
    CHECK(buf[8191] == 0x33);
    CHECK(spdk_mem_get_fd_and_offset(buf, nullptr) == -EINVAL);
    CHECK(munmap(view, 8192) == 0);
    spdk_dma_free(buf);
    CHECK(unregistrations == registrations);
    CHECK(fcntl(fd, F_GETFD) == -1 && errno == EBADF);
    CHECK(spdk_mem_get_fd_and_offset(buf, &offset) == -ENOENT);

    fd = memfd_create("nixl-payload-test", MFD_CLOEXEC);
    CHECK(fd >= 0 && ftruncate(fd, 12288) == 0);
    buf = static_cast<char *>(mmap(nullptr, 8192, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 4096));
    CHECK(buf != MAP_FAILED);
    CHECK(spdk_mem_get_fd_and_offset(buf, &offset) == -ENOENT);
    CHECK(spdk_mem_register(buf, 8192) == -EINVAL);
    CHECK(nixl_spdk_register_shared(buf + 1, 8192) == -EINVAL);
    CHECK(nixl_spdk_register_shared(buf, 12288) == -EINVAL);
    auto *privateBuf = mmap(nullptr, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 4096);
    CHECK(privateBuf != MAP_FAILED);
    memset(privateBuf, 0x66, 8192);
    CHECK(nixl_spdk_register_shared(privateBuf, 8192) == -EINVAL);
    CHECK(munmap(privateBuf, 8192) == 0);
    CHECK(nixl_spdk_register_shared(buf, 8192) == 0);
    CHECK(nixl_spdk_register_shared(buf, 4096) == -EEXIST);
    CHECK(close(fd) == 0);
    fd = spdk_mem_get_fd_and_offset(buf + 8191, &offset);
    CHECK(fd >= 0 && offset == 12287);
    CHECK(spdk_mem_get_fd_and_offset(buf + 8192, &offset) == -ENOENT);
    auto *replay = spdk_mem_map_alloc(0, &ops, nullptr);
    CHECK(replay != nullptr);
    spdk_mem_map_free(&replay);
    view = mmap(nullptr, 8192, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 4096);
    CHECK(view != MAP_FAILED);
    memset(buf, 0xa5, 8192);
    CHECK(memcmp(buf, view, 8192) == 0);
    CHECK(spdk_vtophys(buf, nullptr) == SPDK_VTOPHYS_ERROR);
    CHECK(nixl_spdk_unregister_shared(buf, 8192) == 0);
    CHECK(unregistrations == registrations);
    CHECK(fcntl(fd, F_GETFD) == -1 && errno == EBADF);
    CHECK(munmap(view, 8192) == 0 && munmap(buf, 8192) == 0);
    buf = static_cast<char *>(
        mmap(nullptr, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(buf != MAP_FAILED);
    CHECK(nixl_spdk_register_shared(buf, 8192) == -EINVAL);
    CHECK(munmap(buf, 8192) == 0);
    spdk_mem_map_free(&map);
    spdk_env_fini();
    puts("PASS: FD-backed DMA and imported payloads, notifications, offsets and cleanup");
    return 0;
}
