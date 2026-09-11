/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "s3_control_plane_http.h"

#include <charconv>
#include <exception>
#include <functional>
#include <map>
#include <sstream>
#include <system_error>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSAuthSigner.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/http/URI.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/StringUtils.h>

#include "object/s3/utils.h"
#include "object/s3/aws_sdk_init.h"
#include "common/nixl_log.h"

namespace nixl_obj_rdma {

namespace {
    // Cap on how much of an error response body is echoed into a log line.
    constexpr size_t error_body_log_max = 400;

    // S3 ETag values are returned wrapped in double quotes; strip them.
    std::string
    stripQuotes(const std::string &s) {
        size_t b = 0, e = s.size();
        if (e >= 2 && s.front() == '"' && s.back() == '"') {
            b = 1;
            e = s.size() - 1;
        }
        return s.substr(b, e - b);
    }

    // SigV4-sign the request with payload hash "UNSIGNED-PAYLOAD". A free
    // function (not a member): it derives everything from the request plus the
    // resolved credentials/region, so it has no dependency on the control plane's
    // internals. We sign manually (rather than via the SDK's AWSAuthV4Signer)
    // because that signer hashes the empty body over plain HTTP — the S3 RDMA
    // server only skips content-sha256 validation when the header is exactly
    // UNSIGNED-PAYLOAD, and the data here travels out-of-band over RDMA. This
    // mirrors the standard S3 SigV4 signing. All non-signed headers (host,
    // x-amz-rdma-token, content-*, checksum) must already be set on the request
    // before calling.
    void
    signV4(Aws::Http::HttpRequest &req,
           const Aws::String &access_key,
           const Aws::String &secret_key,
           const Aws::String &session_token,
           const Aws::String &region) {
        using Aws::Utils::HashingUtils;
        using Aws::Utils::StringUtils;
        const Aws::String service = "s3";
        const Aws::String payload_hash = "UNSIGNED-PAYLOAD";

        Aws::Utils::DateTime now = Aws::Utils::DateTime::Now();
        const Aws::String amz_date = now.ToGmtString("%Y%m%dT%H%M%SZ");
        const Aws::String date_stamp = now.ToGmtString("%Y%m%d");

        // Host header (with port) must be signed and match what is sent. Only the
        // scheme's own default port is omitted; a non-default explicit port (even
        // 80 on https or 443 on http) must appear so the signed Host matches the
        // wire, otherwise SigV4 verification fails.
        Aws::String host = req.GetUri().GetAuthority();
        const unsigned p = req.GetUri().GetPort();
        const bool default_port =
            (req.GetUri().GetScheme() == Aws::Http::Scheme::HTTP && p == 80) ||
            (req.GetUri().GetScheme() == Aws::Http::Scheme::HTTPS && p == 443);
        if (p != 0 && !default_port) {
            host += ":" + std::to_string(p);
        }
        req.SetHeaderValue("host", host);
        req.SetHeaderValue("x-amz-date", amz_date);
        req.SetHeaderValue("x-amz-content-sha256", payload_hash);
        if (!session_token.empty()) {
            req.SetHeaderValue("x-amz-security-token", session_token);
        }

        // Canonical headers: lowercase name, trimmed value, sorted by name.
        std::map<Aws::String, Aws::String> hdrs;
        for (const auto &h : req.GetHeaders()) {
            hdrs[StringUtils::ToLower(h.first.c_str())] = StringUtils::Trim(h.second.c_str());
        }
        Aws::String canonical_headers, signed_headers;
        for (const auto &kv : hdrs) {
            canonical_headers += kv.first + ":" + kv.second + "\n";
            if (!signed_headers.empty()) {
                signed_headers += ";";
            }
            signed_headers += kv.first;
        }

        // Canonical query string: sorted, RFC3986-encoded key=value.
        const auto qp = req.GetUri().GetQueryStringParameters();
        std::map<Aws::String, Aws::String> q(qp.begin(), qp.end());
        Aws::String canonical_query;
        for (const auto &kv : q) {
            if (!canonical_query.empty()) {
                canonical_query += "&";
            }
            canonical_query += StringUtils::URLEncode(kv.first.c_str()) + "=" +
                StringUtils::URLEncode(kv.second.c_str());
        }

        Aws::String canonical_uri = req.GetUri().GetURLEncodedPathRFC3986();
        if (canonical_uri.empty()) {
            canonical_uri = "/";
        }

        const Aws::String method =
            Aws::Http::HttpMethodMapper::GetNameForHttpMethod(req.GetMethod());
        const Aws::String canonical_request = method + "\n" + canonical_uri + "\n" +
            canonical_query + "\n" + canonical_headers + "\n" + signed_headers + "\n" +
            payload_hash;

        const Aws::String scope = date_stamp + "/" + region + "/" + service + "/aws4_request";
        const Aws::String cr_hash =
            HashingUtils::HexEncode(HashingUtils::CalculateSHA256(canonical_request));
        const Aws::String string_to_sign =
            "AWS4-HMAC-SHA256\n" + amz_date + "\n" + scope + "\n" + cr_hash;

        auto hmac = [](const Aws::Utils::ByteBuffer &key, const Aws::String &data) {
            return HashingUtils::CalculateSHA256HMAC(
                Aws::Utils::ByteBuffer(reinterpret_cast<const unsigned char *>(data.c_str()),
                                       data.size()),
                key);
        };
        const Aws::String k_secret_str = "AWS4" + secret_key;
        Aws::Utils::ByteBuffer k_secret(
            reinterpret_cast<const unsigned char *>(k_secret_str.c_str()), k_secret_str.size());
        Aws::Utils::ByteBuffer k_date = hmac(k_secret, date_stamp);
        Aws::Utils::ByteBuffer k_region = hmac(k_date, region);
        Aws::Utils::ByteBuffer k_service = hmac(k_region, service);
        Aws::Utils::ByteBuffer k_signing = hmac(k_service, "aws4_request");
        const Aws::String signature = HashingUtils::HexEncode(hmac(k_signing, string_to_sign));

        req.SetHeaderValue("authorization",
                           "AWS4-HMAC-SHA256 Credential=" + access_key + "/" + scope +
                               ", SignedHeaders=" + signed_headers + ", Signature=" + signature);
    }
} // namespace

// ---------------------------------------------------------------------------
// S3RdmaControlPlane
//
// Everything in Impl touches the AWS SDK low-level HTTP/signing layer: SigV4
// UNSIGNED-PAYLOAD signing, URI construction (path vs virtual addressing), and
// response-header retrieval. The surrounding protocol logic (rdmaPut/rdmaGet
// below) is SDK-agnostic and unit tested via rdma_protocol.h.
// ---------------------------------------------------------------------------

struct S3RdmaControlPlane::Impl {
    Aws::String scheme; // "http" / "https"
    Aws::String host; // endpoint host (no port; GetAuthority strips it)
    unsigned port = 0; // explicit port (0 => scheme default)
    Aws::String region;
    bool virtual_addressing = false;
    std::shared_ptr<Aws::Http::HttpClient> http;
    Aws::String access_key;
    Aws::String secret_key;
    Aws::String session_token;

    // Build the token-carrying control-plane request, apply the op-specific
    // headers, SigV4-sign it, and send it. The common prologue (rdma token +
    // content-sha256 sentinel) is shared by PUT and GET; @p set_op_headers adds
    // whatever else the op needs (content-length/checksum for PUT, range for
    // GET) before signing, since the signer canonicalizes the final header set.
    std::shared_ptr<Aws::Http::HttpResponse>
    sendRdmaRequest(Aws::Http::HttpMethod method,
                    const Aws::Http::URI &uri,
                    const char *token,
                    const std::function<void(Aws::Http::HttpRequest &)> &set_op_headers) const {
        auto req = Aws::Http::CreateHttpRequest(
            uri, method, Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
        req->SetHeaderValue("x-amz-content-sha256", unsigned_payload);
        // The token is the RDMA descriptor verbatim; addr/size are its own
        // leading fields, so it is sent as-is (no append).
        req->SetHeaderValue(amz_rdma_token, token);
        set_op_headers(*req);
        signV4(*req, access_key, secret_key, session_token, region);
        return http->MakeRequest(req);
    }

    // Build the request URI for a given object key, applying path-style or
    // virtual-hosted-style addressing.
    Aws::Http::URI
    buildUri(const std::string &bucket, const std::string &key) const {
        Aws::Http::URI uri;
        uri.SetScheme(scheme == "http" ? Aws::Http::Scheme::HTTP : Aws::Http::Scheme::HTTPS);
        if (virtual_addressing) {
            uri.SetAuthority(Aws::String(bucket.c_str()) + "." + host);
            uri.SetPath("/" + Aws::String(key.c_str()));
        } else {
            uri.SetAuthority(host);
            uri.SetPath("/" + Aws::String(bucket.c_str()) + "/" + Aws::String(key.c_str()));
        }
        // GetAuthority() drops the port, so set it explicitly — otherwise the
        // request goes to the scheme default (80/443) and fails to connect.
        if (port != 0) {
            uri.SetPort(static_cast<uint16_t>(port));
        }
        return uri;
    }
};

S3RdmaControlPlane::S3RdmaControlPlane(const nixl_b_params_t *custom_params) : impl_(new Impl()) {
    try {
        nixl_s3_utils::initAWSSDK();

        Aws::Client::ClientConfiguration config;
        nixl_s3_utils::configureClientCommon(config, custom_params);

        impl_->region = config.region.empty() ? Aws::String("us-east-1") : config.region;
        impl_->scheme = (config.scheme == Aws::Http::Scheme::HTTP) ? "http" : "https";
        impl_->virtual_addressing = nixl_s3_utils::getUseVirtualAddressing(custom_params);

        // Endpoint authority: explicit override (an S3-compatible endpoint) or the
        // default AWS S3 regional host.
        if (!config.endpointOverride.empty()) {
            Aws::Http::URI ep(config.endpointOverride);
            // Honor the override's scheme for both http and https (a bare http
            // config.scheme must not stick when the override is https).
            const Aws::String ep_scheme =
                (ep.GetScheme() == Aws::Http::Scheme::HTTP) ? "http" : "https";
            if (ep_scheme != impl_->scheme) {
                NIXL_WARN << "S3 RDMA endpoint_override scheme '" << ep_scheme
                          << "' overrides the configured scheme '" << impl_->scheme << "'";
            }
            impl_->scheme = ep_scheme;
            impl_->host = ep.GetAuthority();
            impl_->port = ep.GetPort();
        } else {
            impl_->host = "s3." + impl_->region + ".amazonaws.com";
            impl_->port = (impl_->scheme == "http") ? 80 : 443;
        }

        // Resolve credentials once (explicit params, else the default chain) and
        // store them for manual SigV4 signing (see signV4).
        //
        // KNOWN LIMITATION: credentials are captured at construction and not
        // refreshed. A long-lived backend using rotating/temporary IAM session
        // tokens will eventually sign with expired credentials; re-create the
        // backend (or extend this to re-query the provider chain per request) to
        // pick up rotated credentials.
        Aws::Auth::AWSCredentials creds;
        auto explicit_creds = nixl_s3_utils::createAWSCredentials(custom_params);
        if (explicit_creds.has_value()) {
            creds = explicit_creds.value();
        } else {
            creds = Aws::Auth::DefaultAWSCredentialsProviderChain().GetAWSCredentials();
        }
        impl_->access_key = creds.GetAWSAccessKeyId();
        impl_->secret_key = creds.GetAWSSecretKey();
        impl_->session_token = creds.GetSessionToken();

        config.connectTimeoutMs = rdma_connect_timeout_secs * 1000;
        config.requestTimeoutMs = rdma_timeout_secs * 1000;
        impl_->http = Aws::Http::CreateHttpClient(config);

        valid_ = impl_->http != nullptr && !impl_->access_key.empty() && !impl_->secret_key.empty();
    }
    catch (const std::exception &e) {
        NIXL_WARN << "S3 RDMA control plane init failed: " << e.what();
        valid_ = false;
    }
}

S3RdmaControlPlane::~S3RdmaControlPlane() = default;

ssize_t
S3RdmaControlPlane::rdmaPut(S3RdmaClientCtx &ctx, const char *token, uint64_t size) {
    // A 0-byte PUT would mint a 0-length RDMA region and move no data; the RDMA
    // path is meaningless for it. Reject up front — a legitimate zero-byte object
    // PUT must go through the ordinary HTTP path in the caller.
    if (size == 0) {
        NIXL_ERROR << "rdmaPut: zero-size request for key=" << ctx.object;
        return rdma_error;
    }
    try {
        Aws::Http::URI uri = impl_->buildUri(ctx.bucket, ctx.object);
        if (!ctx.uploadId.empty()) {
            if (ctx.partNumber == 0 || ctx.partNumber > s3_max_multipart_part_number) {
                NIXL_ERROR << "rdmaPut: invalid partNumber " << ctx.partNumber << " (expected 1.."
                           << s3_max_multipart_part_number << ") for key=" << ctx.object;
                return rdma_error;
            }
            uri.AddQueryStringParameter("uploadId", ctx.uploadId.c_str());
            uri.AddQueryStringParameter("partNumber", std::to_string(ctx.partNumber).c_str());
        }

        auto resp = impl_->sendRdmaRequest(
            Aws::Http::HttpMethod::HTTP_PUT, uri, token, [&ctx](Aws::Http::HttpRequest &req) {
                req.SetHeaderValue("content-type", "application/octet-stream");
                req.SetContentLength("0");
                if (!ctx.checksumCrc64nvme.empty()) {
                    req.SetHeaderValue("x-amz-checksum-crc64nvme", ctx.checksumCrc64nvme.c_str());
                }
            });
        if (!resp) {
            NIXL_ERROR << "rdmaPut: MakeRequest returned null for key=" << ctx.object;
            return rdma_error;
        }

        const int http_status = static_cast<int>(resp->GetResponseCode());
        const std::string etag =
            resp->HasHeader("etag") ? stripQuotes(resp->GetHeader("etag").c_str()) : "";

        // Success: the server completed the RDMA_READ and returns a standard
        // HTTP 200 + ETag (the object payload moved out-of-band, so the HTTP body
        // is empty). Matches the reference client implementations.
        if (http_status == 200 && !etag.empty()) {
            ctx.etag = etag;
            if (resp->HasHeader("x-amz-checksum-crc64nvme")) {
                ctx.checksumCrc64nvme = resp->GetHeader("x-amz-checksum-crc64nvme").c_str();
            }
            return static_cast<ssize_t>(size);
        }

        // Only an explicit `x-amz-rdma-reply: 501` is an RDMA decline. Any other
        // non-200 response (including a plain 4xx/5xx that omits the header) is a
        // real failure — return rdma_error so the retry path still runs, rather
        // than misclassifying it as a decline.
        const std::string reply = resp->HasHeader(amz_rdma_reply) ?
            std::string(resp->GetHeader(amz_rdma_reply).c_str()) :
            "";
        std::ostringstream body;
        body << resp->GetResponseBody().rdbuf();
        if (reply == "501") {
            NIXL_ERROR << "rdmaPut declined: http=" << http_status << " x-amz-rdma-reply='" << reply
                       << "' url=" << uri.GetURIString()
                       << " body=" << body.str().substr(0, error_body_log_max)
                       << " key=" << ctx.object;
            return rdma_not_supported;
        }
        NIXL_ERROR << "rdmaPut failed: http=" << http_status << " x-amz-rdma-reply='" << reply
                   << "' key=" << ctx.object
                   << " body=" << body.str().substr(0, error_body_log_max);
        return rdma_error;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "rdmaPut failed: " << e.what();
        return rdma_error;
    }
}

ssize_t
S3RdmaControlPlane::rdmaGet(S3RdmaClientCtx &ctx,
                            const char *token,
                            uint64_t size,
                            uint64_t offset) {
    // A 0-byte transfer would mint/register a 0-length RDMA region; reject it up
    // front rather than issue a meaningless control-plane request.
    if (size == 0) {
        NIXL_ERROR << "rdmaGet: zero-size request for key=" << ctx.object;
        return rdma_error;
    }
    try {
        Aws::Http::URI uri = impl_->buildUri(ctx.bucket, ctx.object);
        // Byte-range fetch bounded to the caller's buffer (server replies 206 for
        // a slice, 200 for a full object). size is guaranteed non-zero above.
        if (offset > UINT64_MAX - (size - 1)) {
            NIXL_ERROR << "rdmaGet: byte-range overflow (offset=" << offset << " size=" << size
                       << ") for key=" << ctx.object;
            return rdma_error;
        }
        const std::string range =
            "bytes=" + std::to_string(offset) + "-" + std::to_string(offset + size - 1);

        auto resp = impl_->sendRdmaRequest(
            Aws::Http::HttpMethod::HTTP_GET, uri, token, [&range](Aws::Http::HttpRequest &req) {
                req.SetHeaderValue("range", range.c_str());
            });
        if (!resp) {
            NIXL_ERROR << "rdmaGet: MakeRequest returned null for key=" << ctx.object;
            return rdma_error;
        }

        // GET is inherently fail-safe: a non-RDMA server omits x-amz-rdma-reply,
        // which parseRdmaReply maps to "declined" (caller errors under
        // accelerated=true).
        const int http_status = static_cast<int>(resp->GetResponseCode());
        const std::string reply = resp->HasHeader(amz_rdma_reply) ?
            std::string(resp->GetHeader(amz_rdma_reply).c_str()) :
            "";
        const int reply_code = parseRdmaReply(reply);
        if (reply_code == static_cast<int>(rdma_not_supported)) {
            return rdma_not_supported;
        }
        // The HTTP status and the RDMA reply must agree before we trust the
        // transfer: 200 pairs with a full-object success, 206 with a ranged
        // partial. A mismatch (e.g. an error status carrying a stale reply) is a
        // failure.
        const bool accepted = (http_status == 200 && reply_code == rdma_reply_success) ||
            (http_status == 206 && reply_code == rdma_reply_partial_content);
        if (!accepted) {
            NIXL_ERROR << "rdmaGet failed: http=" << http_status << " x-amz-rdma-reply='" << reply
                       << "' reply_code=" << reply_code << " key=" << ctx.object;
            return rdma_error;
        }

        const std::string etag =
            resp->HasHeader("etag") ? stripQuotes(resp->GetHeader("etag").c_str()) : "";

        // The server reports the actual transferred byte count in
        // x-amz-rdma-bytes-transferred: it is authoritative and already clamped
        // to the servable object/range length, so it can be < requested for a
        // ranged/partial GET or an oversized buffer. The header is present iff
        // that count is > 0, so its absence on an accepted 200/206 means a
        // zero-byte transfer (an empty object or range) — report 0, not the
        // requested size. Publish ctx.etag only once the response is fully
        // accepted, so a malformed byte count doesn't leave a stale ETag.
        ssize_t transferred = 0;
        if (resp->HasHeader(amz_rdma_bytes_transferred)) {
            const std::string hdr = resp->GetHeader(amz_rdma_bytes_transferred).c_str();
            uint64_t n = 0;
            const char *begin = hdr.data();
            const char *end = begin + hdr.size();
            auto [parsed_end, ec] = std::from_chars(begin, end, n);
            if (ec != std::errc{} || parsed_end != end || n > size) {
                NIXL_ERROR << "rdmaGet: invalid x-amz-rdma-bytes-transferred='" << hdr
                           << "' (requested " << size << ") for key=" << ctx.object;
                return rdma_error;
            }
            transferred = static_cast<ssize_t>(n);
        }
        ctx.etag = etag;
        return transferred;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "rdmaGet failed: " << e.what();
        return rdma_error;
    }
}

} // namespace nixl_obj_rdma
