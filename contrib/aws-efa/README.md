# NIXL AWS Testing

This directory contains scripts and configuration files for running NIXL tests on AWS infrastructure using AWS Batch and EKS.

## Overview

The AWS test infrastructure allows NIXL to be automatically tested on AWS Elastic Fabric Adapter (EFA) environments through GitHub Actions. It leverages AWS Batch to manage compute resources and job execution.

## Prerequisites

- AWS account with access to EKS and AWS Batch
- Pre-configured AWS job queue: `ucx-nxil-jq`
- Pre-configured AWS EKS cluster: `ucx-ci`
- Properly registered job definition: `NIXL-Ubuntu-JD`

## Files

- **aws_test.sh**: Main script that submits and monitors AWS Batch jobs
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

To run the tests manually:

1. Set your AWS account using `aws configure` command or env vars.
2. Substitute GH variables:

```bash
# Branch or commit SHA to test
export GITHUB_REF="main"

# Other required variables
export GITHUB_SERVER_URL="https://github.com"
export GITHUB_REPOSITORY="ai-dynamo/nixl"

# Run the script with your test command(s)
./aws_test.sh ".gitlab/test_cpp.sh /opt/nixl"

# Multiple test commands can be chained with '&&'
./aws_test.sh ".gitlab/test_cpp.sh /opt/nixl && .test_script2.sh param123"
```

## Test Execution Flow

The AWS test script:

1. Generates AWS job configuration from template
2. Submits a job to AWS
3. Monitors AWS job execution
4. Streams logs from the AWS pod
5. Reports success or failure

## Container Image

The script uses the container image: `nvcr.io/nvidia/pytorch:25.02-py3`
You can override this by setting the `CONTAINER_IMAGE` environment variable:

```bash
export CONTAINER_IMAGE="your-custom-image:tag"
```