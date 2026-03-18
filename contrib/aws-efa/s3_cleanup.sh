#!/bin/bash
set -exE -o pipefail

if [ -z "$NIXL_AWS_ACCESS_KEY_ID" ] || [ -z "$NIXL_AWS_SECRET_ACCESS_KEY" ]; then
    echo "Missing NIXL S3 credentials"
    exit 1
fi

export AWS_ACCESS_KEY_ID="$NIXL_AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$NIXL_AWS_SECRET_ACCESS_KEY"
export AWS_DEFAULT_BUCKET="nixl-ci-test-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"

aws s3 rb "s3://${AWS_DEFAULT_BUCKET}" --force || true

#!/bin/bash
set -x

export AWS_ACCESS_KEY_ID="${NIXL_AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${NIXL_AWS_SECRET_ACCESS_KEY}"

if [ -n "$AWS_DEFAULT_BUCKET" ]; then
    aws s3 rb s3://${AWS_DEFAULT_BUCKET} --force || true
fi
