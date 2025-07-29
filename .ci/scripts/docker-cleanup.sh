#!/bin/bash
set -euo pipefail

# Docker Cleanup Script - Comprehensive residue detection and cleanup
# Usage: docker-cleanup.sh [BUILD_ID] [AXIS_INDEX]

BUILD_ID="${1:-unknown}"
AXIS_INDEX="${2:-unknown}"
SCRIPT_NAME="docker-cleanup.sh"

echo "=== $SCRIPT_NAME: Starting comprehensive Docker cleanup ==="
echo "Build ID: $BUILD_ID, Axis: $AXIS_INDEX"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

# Function to report findings
report_finding() {
    local category="$1"
    local count="$2"
    local details="$3"

    if [ "$count" -gt 0 ]; then
        echo "FOUND: $category - $count items"
        echo "$details"
        echo "---"
    else
        echo "CLEAN: $category - no residues found"
    fi
}

# Function to safe cleanup with error handling
safe_cleanup() {
    local cmd="$1"
    local desc="$2"

    echo "Executing: $desc"
    if eval "$cmd" 2>/dev/null; then
        echo "SUCCESS: $desc"
    else
        echo "WARNING: Failed to execute: $desc (exit code: $?)"
    fi
}

echo
echo "=== PHASE 1: DETECTION ==="

# 1. Check for nixl-ci-test containers (all states)
echo "1. Checking for nixl-ci-test containers..."
CONTAINERS=$(docker ps -a --filter "name=nixl-ci-test" --format "{{.ID}} {{.Names}} {{.Status}}" 2>/dev/null || echo "")
CONTAINER_COUNT=$(echo "$CONTAINERS" | grep -c . || echo "0")
report_finding "nixl-ci-test containers" "$CONTAINER_COUNT" "$CONTAINERS"

# 2. Check for nixl-ci-test images
echo "2. Checking for nixl-ci-test images..."
IMAGES=$(docker images --filter "reference=nixl-ci-test*" --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}" 2>/dev/null || echo "")
IMAGE_COUNT=$(echo "$IMAGES" | grep -c . || echo "0")
report_finding "nixl-ci-test images" "$IMAGE_COUNT" "$IMAGES"

# 3. Check for dangling images
echo "3. Checking for dangling images..."
DANGLING=$(docker images -f "dangling=true" --format "{{.ID}} {{.CreatedAt}}" 2>/dev/null || echo "")
DANGLING_COUNT=$(echo "$DANGLING" | grep -c . || echo "0")
report_finding "dangling images" "$DANGLING_COUNT" "$DANGLING"

# 4. Check for orphaned volumes
echo "4. Checking for orphaned volumes..."
VOLUMES=$(docker volume ls -f "dangling=true" --format "{{.Name}} {{.Driver}}" 2>/dev/null || echo "")
VOLUME_COUNT=$(echo "$VOLUMES" | grep -c . || echo "0")
report_finding "orphaned volumes" "$VOLUME_COUNT" "$VOLUMES"

# 5. Check for dead containers
echo "5. Checking for dead containers..."
DEAD_CONTAINERS=$(docker ps -a --filter "status=dead" --format "{{.ID}} {{.Names}} {{.CreatedAt}}" 2>/dev/null || echo "")
DEAD_COUNT=$(echo "$DEAD_CONTAINERS" | grep -c . || echo "0")
report_finding "dead containers" "$DEAD_COUNT" "$DEAD_CONTAINERS"

# 6. Check for exited containers (potential zombies)
echo "6. Checking for old exited containers..."
EXITED_CONTAINERS=$(docker ps -a --filter "status=exited" --filter "name=nixl-ci-test" --format "{{.ID}} {{.Names}} {{.Status}}" 2>/dev/null || echo "")
EXITED_COUNT=$(echo "$EXITED_CONTAINERS" | grep -c . || echo "0")
report_finding "exited nixl-ci-test containers" "$EXITED_COUNT" "$EXITED_CONTAINERS"

# 7. Check Docker daemon health
echo "7. Checking Docker daemon status..."
DOCKER_INFO=$(docker info --format "{{.Containers}} containers, {{.Images}} images" 2>/dev/null || echo "DOCKER_DAEMON_ERROR")
echo "Docker status: $DOCKER_INFO"

# 8. Check for build cache
echo "8. Checking Docker build cache..."
BUILD_CACHE=$(docker system df --format "{{.Type}}: {{.Size}}" 2>/dev/null | grep -i cache || echo "")
BUILD_CACHE_COUNT=$(echo "$BUILD_CACHE" | grep -c . || echo "0")
report_finding "build cache" "$BUILD_CACHE_COUNT" "$BUILD_CACHE"

echo
echo "=== PHASE 2: CLEANUP ==="

# Cleanup Phase 1: Specific nixl-ci-test resources
if [ "$CONTAINER_COUNT" -gt 0 ]; then
    echo "Cleaning up nixl-ci-test containers..."
    safe_cleanup "docker ps -aq --filter 'name=nixl-ci-test' | xargs -r docker rm -f" "Remove all nixl-ci-test containers"
fi

if [ "$IMAGE_COUNT" -gt 0 ]; then
    echo "Cleaning up nixl-ci-test images..."
    safe_cleanup "docker images -q --filter 'reference=nixl-ci-test*' | xargs -r docker rmi -f" "Remove all nixl-ci-test images"
fi

# Cleanup Phase 2: General Docker cleanup
if [ "$DEAD_COUNT" -gt 0 ]; then
    echo "Cleaning up dead containers..."
    safe_cleanup "docker container prune -f" "Remove dead containers"
fi

if [ "$DANGLING_COUNT" -gt 0 ]; then
    echo "Cleaning up dangling images..."
    safe_cleanup "docker image prune -f" "Remove dangling images"
fi

if [ "$VOLUME_COUNT" -gt 0 ]; then
    echo "Cleaning up orphaned volumes..."
    safe_cleanup "docker volume prune -f" "Remove orphaned volumes"
fi

# Cleanup Phase 3: Network cleanup (if needed)
echo "Cleaning up unused networks..."
safe_cleanup "docker network prune -f" "Remove unused networks"

# Final verification
echo
echo "=== PHASE 3: VERIFICATION ==="
echo "Final container count:"
docker ps -a --filter "name=nixl-ci-test" --format "table {{.Names}}\t{{.Status}}" 2>/dev/null || echo "No containers found"

echo "Final image count:"
docker images --filter "reference=nixl-ci-test*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" 2>/dev/null || echo "No images found"

# System summary
echo
echo "=== SYSTEM SUMMARY ==="
docker system df 2>/dev/null || echo "Could not get system usage"

echo
echo "=== $SCRIPT_NAME: Cleanup completed at $(date '+%Y-%m-%d %H:%M:%S') ==="
