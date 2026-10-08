#pragma once

#include "compiler.hpp"
#include "include_parser.hpp"
#include "kernel_runtime.hpp"

namespace nixl_ep_v2::jit {

static void init(const std::string& library_root_path,
                 const std::string& cuda_home_path_by_python, const std::string& nixl_root_path_by_python, const std::string& ucx_root_path_by_python) {
    Compiler::prepare_init(library_root_path, cuda_home_path_by_python, nixl_root_path_by_python, ucx_root_path_by_python);
    KernelRuntime::prepare_init(cuda_home_path_by_python);
    IncludeParser::prepare_init(library_root_path);
}

static void register_apis(pybind11::module_& m) {
    m.def("init_jit", &init);
}

}  // namespace nixl_ep_v2::jit
