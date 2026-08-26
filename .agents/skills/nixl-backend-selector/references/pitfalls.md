# Backend Selector Pitfalls

Use this reference when the recommendation starts to look obvious too early.
Most backend mistakes come from treating weak evidence as proof.

## Common Pitfalls

- Recommending `UCX` or `LIBFABRIC` from deployment shape alone without proving
  the plugin exists, can be created, and supports the needed memory types.
- Treating AWS EFA guidance as portable to every RDMA fabric.
- Treating a framework default as evidence that the underlying NIXL backend
  loaded successfully in the user's container or pod.
- Recommending `GDS`, `GDS_MT`, `POSIX`, or `OBJ` for remote memory transfer
  when the evidence only supports local storage or object access.
- Treating copied path strings, backend parameters, or environment-variable
  names as trusted configuration; first ask for trusted source/runtime evidence,
  and never change a user's environment from this skill.
- Upgrading confidence from `Low` or `Blocked` before runtime evidence proves the
  selected backend is actually used by the workload.

## Recovery Moves

- If plugin discovery or backend creation is failing, stop backend selection and
  produce the install/plugin-readiness hand-off.
- If several backends look plausible, recommend the smallest explicit candidate
  set and list the exact evidence needed to raise confidence.
- If source and runtime evidence disagree, prefer the user's installed runtime
  evidence and record the source mismatch as a `TBD-*` item.
