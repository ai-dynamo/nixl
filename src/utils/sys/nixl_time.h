#ifndef _NIXL_TIME_H
#define _NIXL_TIME_H

#include <chrono>



namespace nixlTime {

    using namespace std::chrono;

    typedef uint64_t ns_t;
    typedef uint64_t us_t;
    typedef uint64_t ms_t;
    typedef uint64_t sec_t;

    static inline ns_t getNs() {
        return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    }

    static inline us_t getUs() {
        return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
    }

    static inline ms_t getMs() {
        return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
    }

    static inline sec_t getSec() {
        return duration_cast<seconds>(steady_clock::now().time_since_epoch()).count();
    }

}

#endif