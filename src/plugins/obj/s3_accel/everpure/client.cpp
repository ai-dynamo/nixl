/*
 * Copyright 2026 Everpure, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "client.h"
#include "object/s3/utils.h"
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/core/AmazonWebServiceRequest.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpResponse.h>
#include <absl/strings/str_format.h>
#include <atomic>
#include "common/backend.h"
#include "common/configuration.h"
#include "common/exception.h"
#include "common/nixl_log.h"

namespace {

// Header carrying the bare cuObject descriptor - nothing appended. Default
// below; override via environment variable (see client.h) for other
// cuObject-compatible endpoints.
constexpr char kDefaultRdmaTokenHeader[] = "x-amz-rdma-token";

// Standard AWS SigV4 checksum header - fixed name. The RDMA body is always
// empty, so the value defaults to UNSIGNED-PAYLOAD rather than checksumming
// nothing.
constexpr char kContentSha256Header[] = "x-amz-content-sha256";
constexpr char kDefaultContentSha256Value[] = "UNSIGNED-PAYLOAD";

// A successful RDMA response carries this header; its absence means the
// request fell back to ordinary S3 semantics instead of moving data over
// RDMA.
constexpr char kDefaultRdmaReplyHeader[] = "x-amz-rdma-reply";

// RDMA moves the payload out-of-band, so the HTTP body carries no bytes.
constexpr size_t kRdmaBodyLength = 0;

// An explicitly empty header name is ambiguous (disable a check? clear a
// default?), so it's rejected rather than given a meaning. Leaving the
// variable unset is how a caller gets the default.
std::string
requireNonEmptyIfSet(const std::string &env, const std::string &fallback) {
    const auto opt = nixl::config::getValueOptional<std::string>(env);
    if (!opt) {
        return fallback;
    }
    if (opt->empty()) {
        nixl::throwRuntimeError("Config parameter '", env, "' must not be set to an empty value");
    }
    return *opt;
}

std::string
describeDescriptor(std::string_view rdma_desc) {
    return rdma_desc.empty() ? "<empty>" : std::string(rdma_desc);
}

// Rejects a request up front if the descriptor is missing or there's nothing
// to move; both PUT and GET share this baseline check.
bool
rejectIfNoPayload(std::string_view rdma_desc, size_t data_len, const char *op) {
    if (rdma_desc.empty()) {
        NIXL_ERROR << op << ": rdma_desc is empty, refusing to build request";
        return true;
    }
    if (data_len == 0) {
        NIXL_ERROR << op << ": data_len is 0, refusing to build request";
        return true;
    }
    return false;
}

template <typename OutcomeT>
bool
logOutcome(const char *op, const OutcomeT &outcome) {
    if (outcome.IsSuccess()) {
        NIXL_DEBUG << op << ": completed successfully";
        return true;
    }
    const auto &error = outcome.GetError();
    NIXL_ERROR << absl::StrFormat("%s: failed - %s: %s (HTTP %d)",
                                  op,
                                  error.GetExceptionName().c_str(),
                                  error.GetMessage().c_str(),
                                  static_cast<int>(error.GetResponseCode()));
    return false;
}

// Watches for `reply_header` in the response and reports it via the
// returned flag. Needed because PutObjectResult/GetObjectResult drop any
// header the S3 API model doesn't know about, so by the time the async
// callback runs, the confirmation header is already gone.
std::shared_ptr<std::atomic<bool>>
armRdmaConfirmation(Aws::AmazonWebServiceRequest &request, const std::string &reply_header) {
    auto confirmed = std::make_shared<std::atomic<bool>>(false);
    request.SetHeadersReceivedEventHandler(
        [confirmed, reply_header](const Aws::Http::HttpRequest *, Aws::Http::HttpResponse *response) {
            if (response && response->HasHeader(reply_header.c_str())) {
                confirmed->store(true, std::memory_order_relaxed);
            }
        });
    return confirmed;
}

// Succeeds only if both the transport call and the RDMA confirmation check
// pass.
template <typename OutcomeT>
bool
checkRdmaOutcome(const char *op, const OutcomeT &outcome, const std::atomic<bool> &confirmed) {
    if (!logOutcome(op, outcome)) {
        return false;
    }
    if (!confirmed.load(std::memory_order_relaxed)) {
        NIXL_ERROR << op
                   << ": succeeded over HTTP but the RDMA confirmation header was absent - "
                      "the request likely fell back to non-RDMA S3 semantics (check "
                      "NIXL_EVERPURE_RDMA_TOKEN_HEADER/NIXL_EVERPURE_RDMA_REPLY_HEADER against "
                      "the endpoint's protocol)";
        return false;
    }
    return true;
}

} // namespace

awsS3EverpureClient::awsS3EverpureClient(nixl_b_params_t *custom_params,
                                         std::shared_ptr<Aws::Utils::Threading::Executor> executor)
    : awsS3AccelClient(custom_params, executor),
      rdmaTokenHeader_(
          requireNonEmptyIfSet("NIXL_EVERPURE_RDMA_TOKEN_HEADER", kDefaultRdmaTokenHeader)),
      contentSha256Value_(nixl::getBackendParamDefaulted(
          custom_params, "content_sha256_value", std::string(kDefaultContentSha256Value))),
      rdmaReplyHeader_(
          requireNonEmptyIfSet("NIXL_EVERPURE_RDMA_REPLY_HEADER", kDefaultRdmaReplyHeader)) {
    // Own connection pool/timeout budget, separate from the default S3
    // engine's - override via environment variable, else the AWS SDK
    // defaults apply.
    Aws::Client::ClientConfiguration config;
    nixl_s3_utils::configureClientCommon(config, custom_params);
    if (executor) config.executor = executor;

    // The RDMA body is empty, so a checksum over it would never match the
    // real payload; WHEN_REQUIRED stops the SDK computing one by default.
    if (!nixl::getBackendParamOptional<std::string>(custom_params, "req_checksum")) {
        config.checksumConfig.requestChecksumCalculation =
            Aws::Client::RequestChecksumCalculation::WHEN_REQUIRED;
    }
    if (!nixl::getBackendParamOptional<std::string>(custom_params, "resp_checksum")) {
        config.checksumConfig.responseChecksumValidation =
            Aws::Client::ResponseChecksumValidation::WHEN_REQUIRED;
    }

    if (const auto opt =
            nixl::config::getValueOptional<size_t>("NIXL_EVERPURE_MAX_CONNECTIONS")) {
        config.maxConnections = *opt;
    }
    if (const auto opt =
            nixl::config::getValueOptional<long>("NIXL_EVERPURE_CONNECT_TIMEOUT_MS")) {
        config.connectTimeoutMs = *opt;
    }
    if (const auto opt =
            nixl::config::getValueOptional<long>("NIXL_EVERPURE_REQUEST_TIMEOUT_MS")) {
        config.requestTimeoutMs = *opt;
    }

    auto credentials_opt = nixl_s3_utils::createAWSCredentials(custom_params);
    bool use_virtual_addressing = nixl_s3_utils::getUseVirtualAddressing(custom_params);
    s3Client_ = credentials_opt.has_value() ?
        std::make_unique<Aws::S3::S3Client>(
            *credentials_opt,
            config,
            Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::RequestDependent,
            use_virtual_addressing) :
        std::make_unique<Aws::S3::S3Client>(
            config,
            Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::RequestDependent,
            use_virtual_addressing);

    const char *req_checksum_str =
        config.checksumConfig.requestChecksumCalculation ==
                Aws::Client::RequestChecksumCalculation::WHEN_REQUIRED ?
            "required" :
            "supported";
    const char *resp_checksum_str =
        config.checksumConfig.responseChecksumValidation ==
                Aws::Client::ResponseChecksumValidation::WHEN_REQUIRED ?
            "required" :
            "supported";
    NIXL_DEBUG << absl::StrFormat(
        "awsS3EverpureClient ready for S3-RDMA (rdma_token_header=%s, "
        "rdma_reply_header=%s, content_sha256_value=%s, req_checksum=%s, "
        "resp_checksum=%s, max_connections=%u, connect_timeout_ms=%ld, "
        "request_timeout_ms=%ld)",
        rdmaTokenHeader_,
        rdmaReplyHeader_,
        contentSha256Value_,
        req_checksum_str,
        resp_checksum_str,
        config.maxConnections,
        config.connectTimeoutMs,
        config.requestTimeoutMs);
}

void
awsS3EverpureClient::putObjectRdmaAsync(std::string_view key,
                                       uintptr_t data_ptr,
                                       size_t data_len,
                                       size_t offset,
                                       std::string_view rdma_desc,
                                       put_object_callback_t callback) {
    NIXL_DEBUG << absl::StrFormat("putObjectRdmaAsync: key=%s ptr=%p len=%zu offset=%zu desc=%s",
                                  std::string(key).c_str(),
                                  reinterpret_cast<void *>(data_ptr),
                                  data_len,
                                  offset,
                                  describeDescriptor(rdma_desc).c_str());

    // RDMA PUT always writes from byte zero - no partial/append path
    // exists, so a nonzero offset is rejected before touching the wire.
    if (offset != 0) {
        NIXL_ERROR << "putObjectRdmaAsync: RDMA PUT requires offset 0, got " << offset;
        callback(false);
        return;
    }

    if (rejectIfNoPayload(rdma_desc, data_len, "putObjectRdmaAsync")) {
        callback(false);
        return;
    }

    Aws::S3::Model::PutObjectRequest request;
    request.WithBucket(bucketName_).WithKey(Aws::String(key));
    request.SetAdditionalCustomHeaderValue(rdmaTokenHeader_, std::string(rdma_desc));
    if (!contentSha256Value_.empty()) {
        request.SetAdditionalCustomHeaderValue(kContentSha256Header, contentSha256Value_);
    }
    request.SetContentLength(kRdmaBodyLength);

    auto rdma_confirmed = armRdmaConfirmation(request, rdmaReplyHeader_);

    s3Client_->PutObjectAsync(
        request,
        [callback, rdma_confirmed](const Aws::S3::S3Client *,
                   const Aws::S3::Model::PutObjectRequest &,
                   const Aws::S3::Model::PutObjectOutcome &outcome,
                   const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
            // Missing confirmation header despite HTTP success means the
            // payload never moved - the endpoint wrote an empty object at
            // this key instead.
            // TODO: DeleteObjectAsync to clean up that stale object; not
            // implemented.
            callback(checkRdmaOutcome("putObjectRdmaAsync", outcome, *rdma_confirmed));
        },
        nullptr);
}

void
awsS3EverpureClient::getObjectRdmaAsync(std::string_view key,
                                       uintptr_t data_ptr,
                                       size_t data_len,
                                       size_t offset,
                                       std::string_view rdma_desc,
                                       get_object_callback_t callback) {
    NIXL_DEBUG << absl::StrFormat("getObjectRdmaAsync: key=%s ptr=%p len=%zu offset=%zu desc=%s",
                                  std::string(key).c_str(),
                                  reinterpret_cast<void *>(data_ptr),
                                  data_len,
                                  offset,
                                  describeDescriptor(rdma_desc).c_str());

    if (rejectIfNoPayload(rdma_desc, data_len, "getObjectRdmaAsync")) {
        callback(false);
        return;
    }

    const size_t last_byte = offset + (data_len - 1);
    if (last_byte < offset) {
        NIXL_ERROR << "getObjectRdmaAsync: offset " << offset << " + len " << data_len
                   << " overflows, refusing to build request";
        callback(false);
        return;
    }

    Aws::S3::Model::GetObjectRequest request;
    request.WithBucket(bucketName_)
        .WithKey(Aws::String(key))
        .WithRange(absl::StrFormat("bytes=%zu-%zu", offset, last_byte));
    request.SetAdditionalCustomHeaderValue(rdmaTokenHeader_, std::string(rdma_desc));
    if (!contentSha256Value_.empty()) {
        request.SetAdditionalCustomHeaderValue(kContentSha256Header, contentSha256Value_);
    }

    auto rdma_confirmed = armRdmaConfirmation(request, rdmaReplyHeader_);

    s3Client_->GetObjectAsync(
        request,
        [callback, rdma_confirmed](const Aws::S3::S3Client *,
                   const Aws::S3::Model::GetObjectRequest &,
                   const Aws::S3::Model::GetObjectOutcome &outcome,
                   const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
            // A fallback here returns the real object in the HTTP body
            // instead of RDMA, but that body is never read - the caller's
            // buffer stays untouched rather than wrong. Nothing to clean
            // up server-side; GET doesn't modify the object.
            callback(checkRdmaOutcome("getObjectRdmaAsync", outcome, *rdma_confirmed));
        },
        nullptr);
}
