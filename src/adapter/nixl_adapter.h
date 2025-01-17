#include <cstdlib>
#include <iostream>
#include <memory>
#include <stack>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nixl.h"
#include "nixl_descriptors.h"
#include "nixl_params.h"
#include "nixl_types.h"

class nixlAdapter {
    public:
        nixlAdapter();
        ~nixlAdapter();

        int             initialize(const char *role, const char *backend);

        uintptr_t       allocateManagedBuffer(size_t length);
        int             freeManagedBuffer(uintptr_t buffer);
        int             transferAndSync(pybind11::bytes src_desc_bytes,
                                        pybind11::bytes target_desc_bytes,
                                        const std::string target_name,
					                    const std::string op);
        int             waitForCompletion();
        int             writeBytesToBuffer(uintptr_t dst_buffer, char *src_ptr, size_t length);
        pybind11::bytes readBytesFromBuffer(uintptr_t src_address,
					                        size_t length);
        pybind11::bytes registerMemory(uintptr_t buffer_addr, size_t len,
				                       const char *type_name);
        int             deregisterMemory(pybind11::bytes desc_list_blob);

        int		        remoteProgress();
	    pybind11::bytes	getNixlMD();
	    int		        loadNixlMD(pybind11::bytes remote_md_bytes);

    private:
        std::unordered_set<char *>      buffer_alloc_list;
        // nixlAgentConfig                 config;
        nixlUcxInitParams               params;
        nixlBackendEngine               *ucx;
        nixlAgent                       *agent;
};
