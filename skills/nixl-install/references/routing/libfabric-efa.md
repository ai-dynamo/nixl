# LIBFABRIC on AWS EFA

## Scope

Use this route when a workload runs on AWS EFA and NIXL LIBFABRIC is expected,
requested, or being evaluated, but appears unavailable, unselected, unable to
initialize, or unverified. AWS alone, an EFA-capable instance type, installed
EFA software, or UCX use is not enough; UCX may be intentional and supported.

## How to check

Treat an explicit EFA attachment, request, or allocation in EC2 or EKS
deployment data as sufficient evidence of the target. Otherwise, passive
workload evidence requires an EFA device with its bound driver. The output of
`fi_info -p efa` can support this evidence, but provider availability alone
does not prove that EFA is attached or used.

Then inspect NIXL and integration configuration or logs for the intended and
effective backend. Do not query instance metadata, credentials, or cloud APIs
solely for routing.

Required routing facts are whether the workload runs on AWS EFA and whether
LIBFABRIC is intended, required, or being evaluated.
