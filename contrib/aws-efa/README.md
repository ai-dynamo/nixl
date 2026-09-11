# NIXL AWS Testing

This directory contains scripts and configuration files for running NIXL tests on AWS infrastructure using AWS Batch and EKS.

## Overview

The AWS test infrastructure allows NIXL to be automatically tested on AWS Elastic Fabric Adapter (EFA) environments through GitHub Actions. It leverages AWS Batch to manage compute resources and job execution.

## Prerequisites

- AWS account with access to EKS and AWS Batch
- Pre-configured AWS job queue: `ucx-nxil-jq`
- Pre-configured AWS EKS cluster: `ucx-ci`
- Properly registered job definition: `NIXL-Ubuntu-JD`

### EFA-Enabled Security Group Requirements

EFA requires security groups with self-referencing rules. Verify your setup:

```bash
# Get cluster security group
CLUSTER_SG=$(aws eks describe-cluster --name ucx-ci --query 'cluster.resourcesVpcConfig.clusterSecurityGroupId' --output text)

# Verify self-referencing inbound rule (all traffic)
aws ec2 describe-security-groups --group-ids $CLUSTER_SG \
  --query 'SecurityGroups[0].IpPermissions[].UserIdGroupPairs[?GroupId==`'$CLUSTER_SG'`].GroupId'

# Verify SSH access (port 22)
aws ec2 describe-security-groups --group-ids $CLUSTER_SG \
  --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`]' --output json

# Verify self-referencing outbound rule (all traffic)
aws ec2 describe-security-groups --group-ids $CLUSTER_SG \
  --query 'SecurityGroups[0].IpPermissionsEgress[].UserIdGroupPairs[?GroupId==`'$CLUSTER_SG'`].GroupId'
```

**Required rules:** Inbound/outbound all traffic from/to same security group + SSH access.
**Reference:** https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html#efa-start-security

## Files

- **aws_test.sh**: Main script that submits and monitors AWS Batch jobs
- **aws_test_remote.sh**: Remote test entrypoint executed inside the AWS pod
- **aws_vars.template**: Template file for AWS Batch job configuration
- **aws_job_def.json**: Job definition (Registered once, for reference only)

## GitHub Actions Integration

The script is designed to run in GitHub Actions when branches are updated.
It relies on the `GITHUB_REF` environment variable to determine which branch or commit to test. The value can be a branch name (e.g., `main`, `feature-xyz`) or a commit SHA.

### Required GitHub Secrets

The following secrets are configured in the GitHub environment:

- `AWS_ACCESS_KEY_ID`: AWS access key with permissions for AWS Batch and EKS
- `AWS_SECRET_ACCESS_KEY`: Corresponding secret access key

## Manual Execution

`./aws_test.sh` runs the unified flow (build + EFA tests + S3 plugin tests). For manual runs, prepare S3 resources before starting and always clean them up afterward.

1. Set AWS account credentials for AWS Batch/EKS (`aws configure` or env vars).
2. Export required GitHub metadata and NIXL S3 credentials.
3. Run S3 setup, then the AWS validation flow.
4. Run cleanup when done (or after failures).

```bash
# Branch or commit SHA to test
export GITHUB_REF="main"

# Required GitHub metadata
export GITHUB_SERVER_URL="https://github.com"
export GITHUB_REPOSITORY="ai-dynamo/nixl"

# Required for S3 prep/cleanup scripts (S3 plugin tests)
export NIXL_AWS_ACCESS_KEY_ID="<nixl-s3-access-key>"
export NIXL_AWS_SECRET_ACCESS_KEY="<nixl-s3-secret-key>"

# Optional (defaults are used if unset)
export AWS_DEFAULT_REGION="eu-central-1"
# export TEST_TIMEOUT=60

# Prepare S3 bucket and inject S3 env vars into aws_vars.template
./s3_prepare.sh

# Run the AWS validation flow (checkout, build, tests)
./aws_test.sh

# Cleanup S3 resources created by s3_prepare.sh (recommended in all cases)
./s3_cleanup.sh
```

## Test Execution Flow

The AWS test script:

1. Generates AWS job configuration from template
2. Submits a job to AWS
3. Monitors AWS job execution
4. Streams logs from the AWS pod
5. Reports success or failure

## Container Image

The script uses the container image: `nvcr.io/nvidia/cuda-dl-base:25.10-cuda13.0-devel-ubuntu24.04`
You can override this by setting the `CONTAINER_IMAGE` environment variable:

```bash
export CONTAINER_IMAGE="your-custom-image:tag"
```