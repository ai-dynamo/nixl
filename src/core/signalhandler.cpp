#include <iostream>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>

#define MAX_BACKTRACE_SIZE 65535

void handle_signal(int sig) {
    void* buffer[MAX_BACKTRACE_SIZE];

    int nptrs = backtrace(buffer, MAX_BACKTRACE_SIZE);
    fprintf(stderr, "Caught signal %d. Generating stack trace:\n", sig);

    backtrace_symbols_fd(buffer, nptrs, STDERR_FILENO);

    signal(sig, SIG_DFL);
    raise(sig);
}

__attribute__((constructor))
void setup_signal_handlers() {
    signal(SIGSEGV, handle_signal);
    signal(SIGABRT, handle_signal);
    signal(SIGFPE, handle_signal);
    signal(SIGILL, handle_signal);
}
