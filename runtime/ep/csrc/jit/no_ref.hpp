#pragma once

namespace nixl_ep_v2::jit {

// The pointer itself is the kernel argument storage; do not take its address when launching.
struct NoRefPtr {
    void* ptr = nullptr;
};

}  // namespace nixl_ep_v2::jit
