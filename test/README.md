# Test Descriptions

Here are all the explained tests in this directory. There are more specific unit tests in src/utils.

- test/agent_example.cpp - Single threaded test of the nixlAgent API
- test/desc_example.cpp - Test of nixl descriptors and DescList
- test/metadata_streamer.cpp - Single or Multi node test of nixl metadata streamer
- test/nixl_test.cpp - Single or Multi node test of nixlAgent API
- test/ucx_backend_test.cpp - Single threaded test of all the ucxBackendEngine functionality
- test/ucx_backend_multi.cpp - Multi threaded test of UCX connection setup/teardown
- test/python/nixl_bindings_test.py - single threaded Python test of nixlAgent, nixlBasicDesc, and nixlDescList python bindings

# NIXL_wrapper python class

To make the NIXL interface more python style, a wrapper class was added on top of python bindings.
- This class can accept list of descriptors in form of list of tuples of (base_addr, len, devID) alongside a memory type (e.g., "DRAM" or "VRAM").
- It also supports directly a list of Tensors and can extract the required information. DLPack might be supported in future too.
- The initialization is also modified to use a python nixl_config, which can be filled externally with desired backends and corresponding parameters if any.
- Few wrapper functions were added which clarify how dynamicity can be done among NIXL agents. Also instead of passing NIXL binding types, strings are passed.

For this wrapper class two tests are included:
- test/nixl_wrapper_test.py - Basic single node test for tuple style descriptors.
- test/blocking_send_recv_example.py - Basic dual node test for tensor style descriptors, as well as how a blocking send/recv operation can be performed.
