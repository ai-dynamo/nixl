/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <csignal>
#include <iostream>
#include <execinfo.h>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <cstring>

#define MAX_BACKTRACE_DEPTH 100

void
print_backtrace() {
    void *buffer[MAX_BACKTRACE_DEPTH];
    int nptrs = backtrace(buffer, MAX_BACKTRACE_DEPTH);
    backtrace_symbols_fd(buffer, nptrs, STDERR_FILENO);
}

void
gdb_signal_handler(int sig) {
    signal(sig, SIG_DFL);

    const char *header = "\n!!! Caught signal. Generating backtrace: !!!\n";
    ssize_t ignored __attribute__((unused)) = write(STDERR_FILENO, header, strlen(header));

    print_backtrace();

    pid_t tid = fork();
    if (tid == 0) {
        // Child process
        char pid_buf[30] = {0};
        sprintf(pid_buf, "%d", getppid());

        char exe_path_buf[1024];
        ssize_t len = readlink("/proc/self/exe", exe_path_buf, sizeof(exe_path_buf) - 1);
        if (len != -1) {
            exe_path_buf[len] = '\0';
        } else {
            strcpy(exe_path_buf, "UNKNOWN_EXE");
        }

        // Replace child process with GDB
        execlp("gdb",
               "gdb",
               "-q",
               exe_path_buf,
               pid_buf,
               "--batch",
               "-ex",
               "thread apply all bt full",
               "-ex",
               "quit",
               (char *)NULL);

        _exit(1);
    } else if (tid > 0) {
        // Parent process
        int status;
        waitpid(tid, &status, 0);
    }

    // Re-raise signal to get core dump
    raise(sig);
}

__attribute__((constructor)) void
setup_gdb_handler() {
    signal(SIGSEGV, gdb_signal_handler);
    signal(SIGABRT, gdb_signal_handler);
    signal(SIGFPE, gdb_signal_handler);
    signal(SIGILL, gdb_signal_handler);
    signal(SIGBUS, gdb_signal_handler);
}
