#!/bin/bash
# Log in to a container registry, retrying transient failures. One-off
# "invalid username/password" rejections and DNS lookup timeouts have failed
# builds whose sibling branches logged in with the same credentials.
#
# Usage:
#   echo "$TOKEN" | .ci/scripts/registry-login.sh <registry> <username>

set -eu

if [ $# -ne 2 ]; then
    echo "Usage: echo \"\$TOKEN\" | $0 <registry> <username>" >&2
    exit 2
fi

registry="$1"
username="$2"
attempts=5
delay=15
# podman login has no response deadline; a registry that accepts the
# connection but never answers would hang the first attempt forever.
attempt_timeout=60

# Read once so every attempt can replay it.
password="$(cat)"

for attempt in $(seq 1 "${attempts}"); do
    rc=0
    printf '%s\n' "${password}" | timeout "${attempt_timeout}" \
        podman login "${registry}" -u "${username}" --password-stdin || rc=$?
    if [ "${rc}" -eq 0 ]; then
        exit 0
    fi
    if [ "${rc}" -eq 124 ]; then
        echo "Login to ${registry} timed out after ${attempt_timeout}s" >&2
    fi
    if [ "${attempt}" -lt "${attempts}" ]; then
        echo "Login to ${registry} failed (attempt ${attempt}/${attempts}), retrying in ${delay}s" >&2
        sleep "${delay}"
    fi
done

echo "Login to ${registry} failed after ${attempts} attempts" >&2
exit 1
