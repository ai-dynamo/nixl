#ifndef __GDS_UTILS_H
#define __GDS_UTILS_H

#include <fcntl.h>
#include <unistd.h>
#include <nixl.h>

#include "cufile.h"

class gdsFileHandle {
public:
    int            fd;
    // -1 means inf size file?
    size_t         size;
    std::string    metadata;
    CUfileHandle_t cu_fhandle;
};

class gdsMemBuf {
public:
    void   *base;
    size_t size;
};

class gdsUtil {
public:
    gdsUtil() {}
    ~gdsUtil() {}
    nixl_status_t registerFileHandle(int fd, size_t size,
                                     std::string metaInfo,
                                     gdsFileHandle& handle);
    nixl_status_t registerBufHandle(void *ptr, size_t size, int flags);
    void deregisterFileHandle(gdsFileHandle& handle);
    nixl_status_t deregisterBufHandle(void *ptr);
    nixl_status_t openGdsDriver();
    void closeGdsDriver();
};

#endif
