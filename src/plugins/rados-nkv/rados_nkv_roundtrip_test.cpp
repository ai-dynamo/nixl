/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Direct-engine round-trip test for the RADOS_NKV backend.
 *
 * Instantiates nixlRadosNkvEngine directly (no nixlAgent), registers a DRAM
 * source buffer and an OBJ_SEG remote descriptor carrying a token sequence in
 * metaInfo (the engine hashes it into the NVMe KV key), then:
 *   1. WRITE (DRAM -> remote)  => KV Store
 *   2. READ  (remote -> DRAM)  => KV Retrieve
 * and verifies the retrieved bytes match the stored bytes.
 *
 * Usage: rados_nkv_roundtrip_test <vfio-user-dir>
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "nixl_descriptors.h"
#include "backend/backend_aux.h"
#include "rados_nkv_backend.h"

namespace {

bool
checkComplete(const nixlRadosNkvEngine &eng, nixlBackendReqH *h) {
    // Shim ops are synchronous; checkXfer should report SUCCESS immediately.
    for (int i = 0; i < 1000; ++i) {
        nixl_status_t s = eng.checkXfer(h);
        if (s == NIXL_SUCCESS) {
            return true;
        }
        if (s != NIXL_IN_PROG) {
            std::cerr << "checkXfer error status=" << s << "\n";
            return false;
        }
    }
    std::cerr << "checkXfer never completed\n";
    return false;
}

} // namespace

int
main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <vfio-user-dir>\n";
        return 2;
    }
    const std::string vfu_dir = argv[1];

    nixl_b_params_t params;
    params["vfu_addr"] = vfu_dir;
    // Standalone test: no host owns the SPDK env, so this engine brings it up.
    params["init_env"] = "true";

    nixlBackendInitParams init{};
    init.localAgent = "rados_nkv_test_agent";
    init.type = "RADOS_NKV";
    init.customParams = &params;
    init.enableProgTh = false;
    init.pthrDelay = 0;
    init.enableTelemetry_ = false;

    nixlRadosNkvEngine eng(&init);
    if (eng.getInitErr()) {
        std::cerr << "FAIL: engine init error (shim open failed)\n";
        return 1;
    }
    std::cout << "engine initialized against " << vfu_dir << "\n";

    // --- Source DRAM buffer ---
    const std::string payload = "RADOS_NKV-NIXL-roundtrip-slice2-0123456789";
    std::vector<uint8_t> src(payload.begin(), payload.end());
    std::vector<uint8_t> dst(src.size(), 0);

    const uint64_t dram_dev = 0;
    const uint64_t key_dev = 1;
    const std::string kv_key = "nixl-key-0000001"; // 16 bytes

    // Register the local DRAM region (source).
    nixlBlobDesc dram_desc(reinterpret_cast<uintptr_t>(src.data()), src.size(), dram_dev, "");
    nixlBackendMD *dram_md = nullptr;
    if (eng.registerMem(dram_desc, DRAM_SEG, dram_md) != NIXL_SUCCESS) {
        std::cerr << "FAIL: registerMem(DRAM) failed\n";
        return 1;
    }

    // Register the remote OBJ_SEG descriptor carrying the KV key in metaInfo.
    nixlBlobDesc key_desc(0, src.size(), key_dev, kv_key);
    nixlBackendMD *key_md = nullptr;
    if (eng.registerMem(key_desc, OBJ_SEG, key_md) != NIXL_SUCCESS) {
        std::cerr << "FAIL: registerMem(OBJ key) failed\n";
        return 1;
    }
    std::cout << "registered DRAM source and KV key '" << kv_key << "'\n";

    // Build meta dlists for the transfer.
    nixl_meta_dlist_t local(DRAM_SEG);
    local.addDesc(
        nixlMetaDesc(reinterpret_cast<uintptr_t>(src.data()), src.size(), dram_dev, dram_md));

    nixl_meta_dlist_t remote(OBJ_SEG);
    remote.addDesc(nixlMetaDesc(0, src.size(), key_dev, key_md));

    nixl_meta_dlist_t local_dst(DRAM_SEG);
    local_dst.addDesc(
        nixlMetaDesc(reinterpret_cast<uintptr_t>(dst.data()), dst.size(), dram_dev, dram_md));

    // --- WRITE => KV Store ---
    {
        nixlBackendReqH *h = nullptr;
        if (eng.prepXfer(NIXL_WRITE, local, remote, init.localAgent, h) != NIXL_SUCCESS) {
            std::cerr << "FAIL: prepXfer(WRITE) failed\n";
            return 1;
        }
        nixl_status_t ps = eng.postXfer(NIXL_WRITE, local, remote, init.localAgent, h);
        if (ps != NIXL_SUCCESS && ps != NIXL_IN_PROG) {
            std::cerr << "FAIL: postXfer(WRITE) status=" << ps << "\n";
            return 1;
        }
        if (!checkComplete(eng, h)) {
            std::cerr << "FAIL: WRITE did not complete\n";
            return 1;
        }
        eng.releaseReqH(h);
        std::cout << "WRITE (KV Store) of " << src.size() << " bytes complete\n";
    }

    // --- READ => KV Retrieve ---
    {
        nixlBackendReqH *h = nullptr;
        if (eng.prepXfer(NIXL_READ, local_dst, remote, init.localAgent, h) != NIXL_SUCCESS) {
            std::cerr << "FAIL: prepXfer(READ) failed\n";
            return 1;
        }
        nixl_status_t ps = eng.postXfer(NIXL_READ, local_dst, remote, init.localAgent, h);
        if (ps != NIXL_SUCCESS && ps != NIXL_IN_PROG) {
            std::cerr << "FAIL: postXfer(READ) status=" << ps << "\n";
            return 1;
        }
        if (!checkComplete(eng, h)) {
            std::cerr << "FAIL: READ did not complete\n";
            return 1;
        }
        eng.releaseReqH(h);
        std::cout << "READ (KV Retrieve) of " << dst.size() << " bytes complete\n";
    }

    // --- Verify byte-for-byte ---
    if (src != dst) {
        std::cerr << "FAIL: data mismatch after round-trip\n";
        std::cerr << "  wrote: " << payload << "\n";
        std::cerr << "  read : " << std::string(dst.begin(), dst.end()) << "\n";
        return 1;
    }

    std::cout << "verified " << dst.size() << " bytes match: \""
              << std::string(dst.begin(), dst.end()) << "\"\n";

    // --- queryMem => KV Exist (llm-d lookup / cache hit-miss mask) ---
    // A queryMem for the just-Stored key must report PRESENT (engaged optional);
    // a queryMem for a never-stored key must report ABSENT (std::nullopt) and
    // must NOT error (a real miss is not a failure).
    {
        // PRESENT: probe the key we stored above.
        nixl_reg_dlist_t q_present(OBJ_SEG);
        q_present.addDesc(nixlBlobDesc(0, src.size(), key_dev, kv_key));
        std::vector<nixl_query_resp_t> r_present;
        nixl_status_t qs = eng.queryMem(q_present, r_present);
        if (qs != NIXL_SUCCESS) {
            std::cerr << "FAIL: queryMem(present) status=" << qs << "\n";
            return 1;
        }
        if (r_present.size() != 1 || !r_present[0].has_value()) {
            std::cerr << "FAIL: queryMem of stored key '" << kv_key << "' did NOT report PRESENT\n";
            return 1;
        }
        std::cout << "queryMem (KV Exist) of stored key '" << kv_key << "' reports PRESENT (hit)\n";

        // ABSENT: probe a key we never stored. Must be absent, not an error.
        const std::string missing_key = "nixl-key-MISSING"; // 16 bytes, never stored
        nixl_reg_dlist_t q_absent(OBJ_SEG);
        q_absent.addDesc(nixlBlobDesc(0, src.size(), 2, missing_key));
        std::vector<nixl_query_resp_t> r_absent;
        qs = eng.queryMem(q_absent, r_absent);
        if (qs != NIXL_SUCCESS) {
            std::cerr << "FAIL: queryMem(absent) errored (status=" << qs
                      << ") instead of reporting a miss\n";
            return 1;
        }
        if (r_absent.size() != 1 || r_absent[0].has_value()) {
            std::cerr << "FAIL: queryMem of never-stored key '" << missing_key
                      << "' did NOT report ABSENT\n";
            return 1;
        }
        std::cout << "queryMem (KV Exist) of never-stored key '" << missing_key
                  << "' reports ABSENT (miss), not an error\n";
    }

    // --- Key derivation: the KV key is a fixed-length hash of the (arbitrary-
    // length) token sequence in metaInfo. A long token sequence is accepted (no
    // over-length rejection), distinct sequences derive distinct keys, and an
    // empty sequence is rejected. This is a pure check (no Store) so it does not
    // create objects in the backing store. ---
    {
        std::vector<uint8_t> k_short, k_long, k_empty;
        const bool ok_short = radosNkvDeriveKey("nixl-key-0000001", 16, k_short);
        const bool ok_long = radosNkvDeriveKey(std::string(4096, 'x'), 16, k_long);
        if (!ok_short || !ok_long || k_short.size() != 16 || k_long.size() != 16) {
            std::cerr << "FAIL: radosNkvDeriveKey did not produce 16-byte keys\n";
            return 1;
        }
        if (k_short == k_long) {
            std::cerr << "FAIL: distinct token sequences derived the same KV key\n";
            return 1;
        }
        if (radosNkvDeriveKey("", 16, k_empty)) {
            std::cerr << "FAIL: empty token sequence was NOT rejected\n";
            return 1;
        }
        std::cout << "key derivation: arbitrary-length token sequences hash to "
                     "distinct 16-byte keys; empty rejected\n";
    }

    // --- Short-read regression: a READ whose transfer length does not match the
    // stored value length must error, not silently truncate. The value stored
    // above under 'kv_key' is payload.size() bytes; read it back with a
    // deliberately short transfer and require the engine to surface an error. ---
    {
        const size_t short_len = 10; // < payload.size()
        std::vector<uint8_t> short_dst(short_len, 0);

        nixl_meta_dlist_t s_local(DRAM_SEG);
        s_local.addDesc(nixlMetaDesc(
            reinterpret_cast<uintptr_t>(short_dst.data()), short_len, dram_dev, dram_md));
        nixl_meta_dlist_t s_remote(OBJ_SEG);
        s_remote.addDesc(nixlMetaDesc(0, short_len, key_dev, key_md));

        nixlBackendReqH *h = nullptr;
        if (eng.prepXfer(NIXL_READ, s_local, s_remote, init.localAgent, h) != NIXL_SUCCESS) {
            std::cerr << "FAIL: prepXfer(short READ) failed\n";
            return 1;
        }
        nixl_status_t ps = eng.postXfer(NIXL_READ, s_local, s_remote, init.localAgent, h);
        nixl_status_t cs = eng.checkXfer(h);
        eng.releaseReqH(h);
        if (ps == NIXL_SUCCESS || cs == NIXL_SUCCESS) {
            std::cerr << "FAIL: short READ (" << short_len << " bytes) of a " << payload.size()
                      << "-byte value did NOT error (post=" << ps << " check=" << cs << ")\n";
            return 1;
        }
        std::cout << "short READ (" << short_len << " bytes) of a " << payload.size()
                  << "-byte value correctly errored (no silent truncation)\n";
    }

    eng.deregisterMem(key_md);
    eng.deregisterMem(dram_md);

    std::cout << "rados_nkv_roundtrip_test: PASS\n";
    return 0;
}
