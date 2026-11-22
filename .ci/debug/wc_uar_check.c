#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

struct ibv_device;
struct ibv_context;

typedef struct ibv_device **(*get_device_list_t)(int *);
typedef const char *(*get_device_name_t)(struct ibv_device *);

#ifdef IB_FIRST_SHIM

struct ibv_device **
ibv_get_device_list(int *num_devices) {
    static get_device_list_t real_get_list;
    static get_device_name_t get_name;
    const char *first = getenv("IB_FIRST_DEV");
    struct ibv_device **list, *dev;
    void *lib;
    int i;

    if (real_get_list == NULL) {
        lib = dlopen("libibverbs.so.1", RTLD_NOW);
        if (lib == NULL) {
            fprintf(stderr, "ib_first: %s\n", dlerror());
            abort();
        }
        real_get_list = (get_device_list_t)dlsym(lib, "ibv_get_device_list");
        get_name = (get_device_name_t)dlsym(lib, "ibv_get_device_name");
    }

    list = real_get_list(num_devices);
    if ((list == NULL) || (first == NULL) || (*first == '\0')) {
        return list;
    }

    for (i = 0; list[i] != NULL; ++i) {
        if (strcmp(get_name(list[i]), first) == 0) {
            dev = list[i];
            memmove(&list[1], &list[0], i * sizeof(*list));
            list[0] = dev;
            break;
        }
    }

    return list;
}

#else

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;

struct dv_context_attr {
    uint32_t flags;
    uint64_t comp_mask;
    void *fds;
};

struct dv_devx_uar {
    void *reg_addr;
    void *base_addr;
    uint32_t page_id;
    off_t mmap_off;
    uint64_t comp_mask;
};

#define DV_CONTEXT_FLAGS_DEVX 0x1
#define DV_UAR_ALLOC_TYPE_WC 0x0
#define BF_REG_SIZE 256
#define CU_REG_PORTABLE_DEVICEMAP_IOMEMORY 0x7

static void *
open_lib(const char *name) {
    void *lib = dlopen(name, RTLD_NOW | RTLD_GLOBAL);

    if (lib == NULL) {
        fprintf(stderr, "wc_check: %s\n", dlerror());
        exit(1);
    }
    return lib;
}

static void *
load(void *lib, const char *name) {
    void *sym = dlsym(lib, name);

    if (sym == NULL) {
        fprintf(stderr, "wc_check: %s not found\n", name);
        exit(1);
    }
    return sym;
}

static void
read_port_state(const char *dev, char *buf, size_t size) {
    char path[256];
    FILE *file;

    snprintf(buf, size, "?");
    snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/1/state", dev);
    file = fopen(path, "r");
    if (file == NULL) {
        return;
    }
    if (fgets(buf, size, file) != NULL) {
        buf[strcspn(buf, "\n")] = '\0';
    }
    fclose(file);
}

int
main(void) {
    void *ibv = open_lib("libibverbs.so.1");
    void *mlx5 = open_lib("libmlx5.so.1");
    void *cuda = open_lib("libcuda.so.1");
    get_device_list_t get_list = load(ibv, "ibv_get_device_list");
    get_device_name_t get_name = load(ibv, "ibv_get_device_name");
    void (*free_list)(struct ibv_device **) = load(ibv, "ibv_free_device_list");
    int (*close_dev)(struct ibv_context *) = load(ibv, "ibv_close_device");
    struct ibv_context *(*open_dv)(struct ibv_device *, struct dv_context_attr *) =
        load(mlx5, "mlx5dv_open_device");
    struct dv_devx_uar *(*alloc_uar)(struct ibv_context *, uint32_t) =
        load(mlx5, "mlx5dv_devx_alloc_uar");
    void (*free_uar)(struct dv_devx_uar *) = load(mlx5, "mlx5dv_devx_free_uar");
    CUresult (*cu_init)(unsigned) = load(cuda, "cuInit");
    CUresult (*cu_device_get)(CUdevice *, int) = load(cuda, "cuDeviceGet");
    CUresult (*cu_ctx_retain)(CUcontext *, CUdevice) = load(cuda, "cuDevicePrimaryCtxRetain");
    CUresult (*cu_ctx_set)(CUcontext) = load(cuda, "cuCtxSetCurrent");
    CUresult (*cu_register)(void *, size_t, unsigned) = load(cuda, "cuMemHostRegister_v2");
    CUresult (*cu_unregister)(void *) = load(cuda, "cuMemHostUnregister");
    CUresult (*cu_error_string)(CUresult, const char **) = load(cuda, "cuGetErrorString");
    char wc_fail_dev[64] = "", wc_ok_dev[64] = "";
    struct ibv_device **list;
    CUcontext cu_ctx;
    CUdevice cu_dev;
    int i, num;

    if ((cu_init(0) != 0) || (cu_device_get(&cu_dev, 0) != 0) ||
        (cu_ctx_retain(&cu_ctx, cu_dev) != 0) || (cu_ctx_set(cu_ctx) != 0)) {
        fprintf(stderr, "wc_check: CUDA context setup failed\n");
        return 1;
    }

    list = get_list(&num);
    if (list == NULL) {
        fprintf(stderr, "wc_check: ibv_get_device_list: %s\n", strerror(errno));
        return 1;
    }

    for (i = 0; i < num; ++i) {
        struct dv_context_attr attr = {.flags = DV_CONTEXT_FLAGS_DEVX};
        const char *name = get_name(list[i]);
        const char *cu_err = "?";
        char state[64], wc[96], reg[128];
        struct ibv_context *ctx;
        struct dv_devx_uar *uar;
        CUresult res;

        read_port_state(name, state, sizeof(state));
        ctx = open_dv(list[i], &attr);
        if (ctx == NULL) {
            fprintf(stderr, "%-8s port=%-10s open=FAIL(%s)\n", name, state, strerror(errno));
            continue;
        }

        snprintf(reg, sizeof(reg), "-");
        uar = alloc_uar(ctx, DV_UAR_ALLOC_TYPE_WC);
        if (uar == NULL) {
            snprintf(wc, sizeof(wc), "FAIL(%s)", strerror(errno));
            if (wc_fail_dev[0] == '\0') {
                snprintf(wc_fail_dev, sizeof(wc_fail_dev), "%s", name);
            }
        } else {
            snprintf(wc, sizeof(wc), "OK");
            res = cu_register(uar->reg_addr, BF_REG_SIZE, CU_REG_PORTABLE_DEVICEMAP_IOMEMORY);
            if (res == 0) {
                cu_unregister(uar->reg_addr);
                snprintf(reg, sizeof(reg), "OK");
                if ((wc_ok_dev[0] == '\0') && (strstr(state, "ACTIVE") != NULL)) {
                    snprintf(wc_ok_dev, sizeof(wc_ok_dev), "%s", name);
                }
            } else {
                cu_error_string(res, &cu_err);
                snprintf(reg, sizeof(reg), "FAIL(%d %s)", res, cu_err);
            }
            free_uar(uar);
        }

        fprintf(stderr, "%-8s port=%-10s wc_uar=%-30s cuda_register=%s\n", name, state, wc, reg);
        close_dev(ctx);
    }

    free_list(list);
    printf("WC_FAIL_DEV=%s WC_OK_DEV=%s\n", wc_fail_dev, wc_ok_dev);
    return 0;
}

#endif
