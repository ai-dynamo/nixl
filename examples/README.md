# NIXL Examples

This directory contains runnable examples across Python, C++, and Rust.

 The `examples/python/basic_two_peers.py` is a good starting point to learn the basic usage of NIXL end-to-end.
 It shows basic registration, metadata exchange, and transfer operations.

 The `examples/python/expanded_two_peers.py` further extends the basic example with more use cases and demonstrates:
 - Registration best practice: fewer, larger registrations reduce kernel calls and internal lookups
 - Creating descriptors used for transfers in various modes, from tensor, python tuples, numpy
 - Creating transfers in different modes:
  - Prepped flow (prep_xfer_dlist + make_prepped_xfer): when blocks are known in advance, prepare once and choose indices/mappings at runtime
  - Combined flow (initialize_xfer): NIXL prepares and creates the transfer handle in one step; useful when there is no fixed notion of blocks
 - Application-driven ordering vs. parallelism (posting groups and waiting to enforce order)
 - Reposting transfer handles after completion
 - Notification checks, data verification, and explicit teardown at the end
