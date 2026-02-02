#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# shellcheck disable=SC1091
. "$(dirname "$0")/../.ci/scripts/common.sh"

set -e
set -x

export NVM_DIR=${HOME}/.nvm
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

AZ_ACCOUNT_NAME="nixl-ci-dev"
AZ_ACCOUNT_KEY="ZGV2c3RvcmVhY2NvdW50Mw=="   # "devstoreaccount3" base64
export AZURITE_ACCOUNTS="${AZ_ACCOUNT_NAME}:${AZ_ACCOUNT_KEY}"
AZURITE_WORKDIR="./azdata"
mkdir -p "${AZURITE_WORKDIR}"

blob_port=$(get_next_tcp_port)

echo "Starting Azurite blob service with custom account: ${AZ_ACCOUNT_NAME}"

azurite-blob --skipApiVersionCheck -l "${AZURITE_WORKDIR}" --blobHost 127.0.0.1 --blobPort "$blob_port" &

AZURITE_PID=$!
echo "Azurite blob PID: ${AZURITE_PID}"

# Give Azurite a moment to start
sleep 3

# Build a connection string targeting our custom account
CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=${AZ_ACCOUNT_NAME};AccountKey=${AZ_ACCOUNT_KEY};BlobEndpoint=http://127.0.0.1:${blob_port}/${AZ_ACCOUNT_NAME};"

CONTAINER_NAME="testcontainer"
BLOB_NAME="hello.txt"
LOCAL_FILE="hello.txt"
DOWNLOADED_FILE="hello-downloaded.txt"

# Create a test file
echo "Hello from Azurite with custom account!" > "${LOCAL_FILE}"

echo "Creating container: ${CONTAINER_NAME}"
az storage container create \
  --name "${CONTAINER_NAME}" \
  --connection-string "${CONNECTION_STRING}"

echo "Uploading blob: ${BLOB_NAME}"
az storage blob upload \
  --container-name "${CONTAINER_NAME}" \
  --name "${BLOB_NAME}" \
  --file "${LOCAL_FILE}" \
  --overwrite true \
  --connection-string "${CONNECTION_STRING}"

echo "Downloading blob to: ${DOWNLOADED_FILE}"
az storage blob download \
  --container-name "${CONTAINER_NAME}" \
  --name "${BLOB_NAME}" \
  --file "${DOWNLOADED_FILE}" \
  --connection-string "${CONNECTION_STRING}"

echo "Downloaded contents:"
cat "${DOWNLOADED_FILE}"

echo "Stopping Azurite (PID ${AZURITE_PID})"
kill "${AZURITE_PID}" || true
wait "${AZURITE_PID}" 2>/dev/null || true

echo "Done."
