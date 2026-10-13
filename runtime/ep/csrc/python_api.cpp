#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <nixl_ep_v2/common/compiled.cuh>

#include "jit/api.hpp"
#include "elastic/buffer.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "NIXL EP V2: an efficient expert-parallel communication library";

    // Whether support FP8 and TMA features
    m.def("is_sm90_compiled", []() { return nixl_ep_v2::kEnableSM90Features; });
    m.def("get_adaptive_num_launch_sms", []() { return nixl_ep_v2::elastic::topology::get_num_launch_sms(); });
    m.def("is_fabric_handle_supported", &nixl_ep_v2::elastic::nixl::is_fabric_handle_supported);

    // The integer type of top-k indices
    m.attr("topk_idx_t") = py::cast(c10::CppTypeToScalarType<nixl_ep_v2::topk_idx_t>::value);

    // JIT API
    nixl_ep_v2::jit::register_apis(m);

    // Register elastic buffer (NIXL EP V2) APIs
    nixl_ep_v2::elastic::register_apis(m);
}
