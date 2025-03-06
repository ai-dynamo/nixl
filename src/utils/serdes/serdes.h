#ifndef __SERDES_H
#define __SERDES_H

#include <cstring>
#include <string>
#include <cstdint>

#include "nixl_types.h"

class nixlSerDes {
private:
    typedef enum { SERIALIZE, DESERIALIZE } ser_mode_t;

    std::string workingStr;
    ssize_t des_offset;
    ser_mode_t mode;

public:
    nixlSerDes();

    /* Ser/Des for Strings */
    nixl_status_t addStr(const std::string &tag, const std::string &str);
    std::string getStr(const std::string &tag);

    /* Ser/Des for Byte buffers */
    nixl_status_t addBuf(const std::string &tag, const void* buf, ssize_t len);
    ssize_t getBufLen(const std::string &tag) const;
    nixl_status_t getBuf(const std::string &tag, void *buf, ssize_t len);

    /* Ser/Des buffer management */
    std::string exportStr() const;
    nixl_status_t importStr(const std::string &sdbuf);

    static std::string _bytesToString(const void *buf, ssize_t size);
    static void _stringToBytes(void* fill_buf, const std::string &s, ssize_t size);
};

#endif
