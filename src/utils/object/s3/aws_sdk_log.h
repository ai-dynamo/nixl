/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_SRC_UTILS_OBJECT_S3_AWS_SDK_LOG_H
#define NIXL_SRC_UTILS_OBJECT_S3_AWS_SDK_LOG_H

#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/utils/logging/LogLevel.h>
#include <aws/core/utils/logging/CRTLogSystem.h>
#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "common/nixl_log.h"

namespace nixl_s3_utils {

/**
 * Maps the NIXL_LOG_LEVEL environment variable to the corresponding AWS SDK
 * LogLevel so the AWS logger is gated at the same verbosity as NIXL itself.
 * Defaults to Warn if the variable is not set or unrecognized.
 */
inline Aws::Utils::Logging::LogLevel
getNixlAwsLogLevel() {
    using Aws::Utils::Logging::LogLevel;
    const char *env = std::getenv("NIXL_LOG_LEVEL");
    if (!env) return LogLevel::Warn;

    std::string lvl(env);
    for (auto &c : lvl)
        c = std::toupper(static_cast<unsigned char>(c));

    if (lvl == "TRACE") return LogLevel::Trace;
    if (lvl == "DEBUG") return LogLevel::Debug;
    if (lvl == "INFO") return LogLevel::Info;
    if (lvl == "WARN") return LogLevel::Warn;
    if (lvl == "ERROR") return LogLevel::Error;
    if (lvl == "FATAL") return LogLevel::Fatal;
    return LogLevel::Warn;
}

/**
 * AWS SDK logger that routes all SDK messages through NIXL's logging macros.
 *
 * Messages are prefixed with "[AWS:<tag>]" where <tag> is the AWS SDK component
 * name (e.g. "S3Transfer", "AWSS3", "AWSClient") to allow easy filtering.
 *
 * Implements LogSystemInterface directly (rather than FormattedLogSystem) to
 * avoid double-timestamping: FormattedLogSystem prepends its own timestamp and
 * thread ID before the message reaches NIXL/Abseil, which adds another prefix.
 *
 * Log level mapping:
 *   AWS Fatal  ->  NIXL_ERROR  (not NIXL_FATAL: AWS-internal fatals must not abort the process)
 *   AWS Error  ->  NIXL_ERROR
 *   AWS Warn   ->  NIXL_WARN
 *   AWS Info   ->  NIXL_INFO
 *   AWS Debug  ->  NIXL_DEBUG  (VLOG(1))
 *   AWS Trace  ->  NIXL_TRACE  (DVLOG(2), stripped in release builds)
 */
class NixlAwsLogger : public Aws::Utils::Logging::LogSystemInterface {
public:
    explicit NixlAwsLogger(Aws::Utils::Logging::LogLevel level) : level_(level) {}

    Aws::Utils::Logging::LogLevel
    GetLogLevel() const override {
        return level_.load();
    }

    void
    LogStream(Aws::Utils::Logging::LogLevel logLevel,
              const char *tag,
              const Aws::OStringStream &messageStream) override {
        dispatch(logLevel, tag, messageStream.str().c_str());
    }

    void
    Log(Aws::Utils::Logging::LogLevel logLevel,
        const char *tag,
        const char *formatStr,
        ...) override {
        va_list args;
        va_start(args, formatStr);
        vaLog(logLevel, tag, formatStr, args);
        va_end(args);
    }

    void
    vaLog(Aws::Utils::Logging::LogLevel logLevel,
          const char *tag,
          const char *formatStr,
          va_list args) override {
        char buf[4096];
        vsnprintf(buf, sizeof(buf), formatStr, args);
        dispatch(logLevel, tag, buf);
    }

    void
    Flush() override {}

private:
    std::atomic<Aws::Utils::Logging::LogLevel> level_;

    void
    dispatch(Aws::Utils::Logging::LogLevel logLevel, const char *tag, const char *msg) {
        using Aws::Utils::Logging::LogLevel;
        switch (logLevel) {
        // Fatal -> ERROR: AWS-internal fatals must not abort the NIXL process
        case LogLevel::Fatal:
            NIXL_ERROR << "[AWS:" << tag << "] " << msg;
            break;
        case LogLevel::Error:
            NIXL_ERROR << "[AWS:" << tag << "] " << msg;
            break;
        case LogLevel::Warn:
            NIXL_WARN << "[AWS:" << tag << "] " << msg;
            break;
        case LogLevel::Info:
            NIXL_INFO << "[AWS:" << tag << "] " << msg;
            break;
        case LogLevel::Debug:
            NIXL_DEBUG << "[AWS:" << tag << "] " << msg;
            break;
        case LogLevel::Trace:
            NIXL_TRACE << "[AWS:" << tag << "] " << msg;
            break;
        default:
            break;
        }
    }
};

/**
 * CRT logger that routes aws-c-* runtime messages through NIXL's logging macros.
 *
 * The AWS SDK C++ CRT layer (aws-c-s3, aws-c-http, aws-c-common) uses a separate
 * logging interface (CRTLogSystemInterface) from the SDK C++ layer (LogSystemInterface).
 * This class bridges that interface to NIXL macros so that CRT-internal events such as
 * multipart upload progress, connection pool events, and TLS handshakes are visible
 * alongside regular SDK logs.
 *
 * Messages are prefixed with "[CRT:<subject>]" where <subject> is the aws-c-* component
 * name (e.g. "aws-c-s3", "aws-c-http", "aws-c-common") to distinguish them from SDK logs.
 *
 * Registered via SDKOptions::crt_logger_create_fn in initAWSSDK().
 *
 * Log level mapping:
 *   CRT Fatal  ->  NIXL_ERROR  (not NIXL_FATAL: CRT-internal fatals must not abort the process)
 *   CRT Error  ->  NIXL_ERROR
 *   CRT Warn   ->  NIXL_WARN
 *   CRT Info   ->  NIXL_INFO
 *   CRT Debug  ->  NIXL_DEBUG  (VLOG(1))
 *   CRT Trace  ->  NIXL_DEBUG  (VLOG(1); CRT Trace is the normal operational level for S3)
 */
class NixlCrtLogSystem : public Aws::Utils::Logging::CRTLogSystemInterface {
public:
    explicit NixlCrtLogSystem(Aws::Utils::Logging::LogLevel level) : level_(level) {}

    Aws::Utils::Logging::LogLevel
    GetLogLevel() const override {
        return level_.load();
    }

    void
    SetLogLevel(Aws::Utils::Logging::LogLevel level) override {
        level_.store(level);
    }

    void
    Log(Aws::Utils::Logging::LogLevel logLevel,
        const char *subjectName,
        const char *formatStr,
        va_list args) override {
        char buf[4096];
        vsnprintf(buf, sizeof(buf), formatStr, args);

        // Strip trailing newline that aws-c-* appends to most messages
        size_t len = std::strlen(buf);
        while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
            buf[--len] = '\0';

        dispatch(logLevel, subjectName ? subjectName : "?", buf);
    }

private:
    std::atomic<Aws::Utils::Logging::LogLevel> level_;

    void
    dispatch(Aws::Utils::Logging::LogLevel logLevel, const char *tag, const char *msg) {
        using Aws::Utils::Logging::LogLevel;
        switch (logLevel) {
        // Fatal -> ERROR: CRT-internal fatals must not abort the NIXL process
        case LogLevel::Fatal:
            NIXL_ERROR << "[CRT:" << tag << "] " << msg;
            break;
        case LogLevel::Error:
            NIXL_ERROR << "[CRT:" << tag << "] " << msg;
            break;
        case LogLevel::Warn:
            NIXL_WARN << "[CRT:" << tag << "] " << msg;
            break;
        case LogLevel::Info:
            NIXL_INFO << "[CRT:" << tag << "] " << msg;
            break;
        case LogLevel::Debug:
        case LogLevel::Trace:
            NIXL_DEBUG << "[CRT:" << tag << "] " << msg;
            break;
        default:
            break;
        }
    }
};

} // namespace nixl_s3_utils

#endif // NIXL_SRC_UTILS_OBJECT_S3_AWS_SDK_LOG_H
