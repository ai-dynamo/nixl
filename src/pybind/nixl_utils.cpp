#include <pybind11/pybind11.h>

namespace py = pybind11;

//JUST FOR TESTING
uintptr_t malloc_passthru(int size) {
    return (uintptr_t) malloc(size);
}

//JUST FOR TESTING
void free_passthru(uintptr_t buf) {
    free((void*) buf);
}

//JUST FOR TESTING
void ba_buf(uintptr_t addr, int size) {
    uint8_t* buf = (uint8_t*) addr;
    for(int i = 0; i<size; i++) buf[i] = 0xba;
}

//JUST FOR TESTING
void verify_transfer(uintptr_t addr1, uintptr_t addr2, int size) {
    uint8_t* buf1 = (uint8_t*) addr1;
    uint8_t* buf2 = (uint8_t*) addr2;

    for(int i = 0; i<size; i++) assert(buf1[i] == buf2[i]);
}

PYBIND11_MODULE(nixl_utils, m) {
    m.def("malloc_passthru", &malloc_passthru);
    m.def("free_passthru", &free_passthru);
    m.def("ba_buf", &ba_buf);
    m.def("verify_transfer", &verify_transfer);
}
