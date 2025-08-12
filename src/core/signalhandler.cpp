#include <csignal>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <cstring>

void gdb_signal_handler(int sig) {
    signal(sig, SIG_DFL);

    const char* header = "\n!!! Caught signal. Forking GDB for full backtrace: !!!\n";
    write(STDERR_FILENO, header, strlen(header));

    pid_t tid = fork();
    if (tid == 0) { // Child process
        char pid_buf[30] = {0};
        sprintf(pid_buf, "%d", getppid()); // Get parent's PID

        char exe_path_buf[1024];
        ssize_t len = readlink("/proc/self/exe", exe_path_buf, sizeof(exe_path_buf) - 1);
        if (len != -1) {
            exe_path_buf[len] = '\0';
        } else {
            strcpy(exe_path_buf, "UNKNOWN_EXE");
        }

        // Replace child process with GDB
        execlp("gdb", "gdb", "-q", exe_path_buf, pid_buf, "--batch",
               "-ex", "thread apply all bt full", "-ex", "quit",
               (char *)NULL);

        // If execlp fails, we exit
        perror("execlp");
        _exit(1);

    } else if (tid > 0) { // Parent process
        // Wait for GDB to finish printing the backtrace
        int status;
        waitpid(tid, &status, 0);
    }

    // Re-raise signal to get core dump
    raise(sig);
}

__attribute__((constructor))
void setup_gdb_handler() {
    // Only use one handler!
    // signal(SIGSEGV, safe_signal_handler);
    signal(SIGSEGV, gdb_signal_handler);
    signal(SIGABRT, gdb_signal_handler);
    signal(SIGFPE, gdb_signal_handler);
    signal(SIGILL, gdb_signal_handler);
}
