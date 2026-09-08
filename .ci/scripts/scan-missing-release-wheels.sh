#!/bin/bash -eE
# Scan release/* branches for commits whose published wheels are missing this
# CUDA variant, and write triggers.txt (one "<sha> <ver> <cuda_major>" line per
# missing build) for the poller's trigger step.
#
# Env (set by build-wheel-release-poller-matrix.yaml):
#   cuda_major    - CUDA major version (matrix axis); names the variant cuNN
#   MIN_RELEASE   - releases older than this are not built
#   MAX_COMMITS   - newest N first-parent commits to check per release branch
#   NIXL_REPO_URL, STORAGE_API_URL, ARTIFACTORY_USER, ARTIFACTORY_TOKEN

variant="cu${cuda_major}"

# The Jenkins checkout is owned by a different uid; trust only it.
git config --global --add safe.directory "${PWD}"

# The job checkout is the CI config; merge-base/rev-list need local refs for
# main and the release branches.
git fetch --no-tags "${NIXL_REPO_URL}" \
  '+refs/heads/main:refs/remotes/origin/main' \
  '+refs/heads/release/*:refs/remotes/origin/release/*'

: > triggers.txt

branches="$(git for-each-ref --format='%(refname:lstrip=4)' 'refs/remotes/origin/release/*' | sort -V)"

# Keep the Artifactory token out of the log (also under CI debug -x).
set +x

for ver in ${branches}; do
  # Only dotted-numeric release branches; else sort -V mis-ranks names.
  if ! printf '%s' "${ver}" | grep -qE '^[0-9]+(\.[0-9]+)+$'; then
    echo "release/${ver}: not a numeric version, skipping"
    continue
  fi
  if ! printf '%s\n%s\n' "${MIN_RELEASE}" "${ver}" | sort -CV; then
    echo "release/${ver}: below ${MIN_RELEASE}, skipping"
    continue
  fi

  # The nightly always passes --build-options-file (and the plugin flags), so a
  # release whose build-container.sh predates them fails at option parsing.
  # Skip it rather than fan out builds that cannot succeed; release branches cut
  # from main after that change pass on their own.
  if ! git show "origin/release/${ver}:contrib/build-container.sh" 2>/dev/null \
       | grep -q -- '--build-options-file'; then
    echo "release/${ver}: build-container.sh predates --build-options-file, skipping"
    continue
  fi

  base="$(git merge-base origin/main "origin/release/${ver}")"
  candidates="$(git rev-list --first-parent "${base}..origin/release/${ver}" | head -"${MAX_COMMITS}")"

  n_cand=0; n_build=0
  for sha in ${candidates}; do
    folder="release/${ver}/${sha:0:8}"

    # Only 200 (folder listing) and 404 (folder absent) are conclusive;
    # other errors (auth, 5xx, network) skip the commit until the next
    # cycle so an Artifactory hiccup cannot fan out spurious builds.
    http_code="$(curl -s --connect-timeout 10 --max-time 30 -o folder.json -w '%{http_code}' \
      -u "${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}" "${STORAGE_API_URL}/${folder}/")" || http_code=""
    if [ "${http_code}" != "200" ] && [ "${http_code}" != "404" ]; then
      echo "release/${ver}: ${folder}: storage API returned ${http_code:-<none>}, skipping this cycle"
      continue
    fi

    n_cand=$((n_cand+1))
    # The variant is already published when the folder lists a nixl_cuNN wheel.
    if [ "${http_code}" = "200" ] && grep -q "\"/nixl_${variant}-" folder.json; then
      continue
    fi
    echo "${sha} ${ver} ${cuda_major}" >> triggers.txt
    n_build=$((n_build+1))
  done

  echo "release/${ver} (${variant}): candidates=${n_cand} to_build=${n_build}"
done

echo "=== Poller summary (${variant}): $(wc -l < triggers.txt) build(s) to trigger ==="
cat triggers.txt
