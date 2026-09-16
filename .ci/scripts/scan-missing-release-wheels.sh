#!/bin/bash -eE
# Scan release/* branches for commits with no published wheel set, and write
# triggers.txt (one "<sha> <ver>" line per missing build) for the poller's
# trigger step.
#
# A commit counts as published when a build folder under release/<ver>/ carries
# it as a NIXL_SHA property. The wheel job stamps that property on the folder
# only after the whole build goes green (and deletes the folder outright when it
# does not), so the marker means "complete cu12+cu13 wheel set" - which the old
# check, "the sha-named folder lists a nixl_cuNN file", could not distinguish
# from a half-finished upload.
#
# Env (set by build-wheel-release-poller-matrix.yaml):
#   MIN_RELEASE   - releases older than this are not built
#   MAX_COMMITS   - newest N first-parent commits to check per release branch
#   NIXL_REPO_URL, AQL_API_URL, WHEEL_REPO_NAME, ARTIFACTORY_USER, ARTIFACTORY_TOKEN

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

  # One AQL for the whole release instead of one GET per commit: collect the
  # NIXL_SHA marker of every completed build folder under release/<ver>/.
  # repo/path/name are mandatory in any items .include() - Artifactory rejects
  # the query outright without them ("for permissions reasons").
  aql="items.find({\"repo\":\"${WHEEL_REPO_NAME}\",\"type\":\"folder\",\"path\":\"release/${ver}\",\"@NIXL_SHA\":{\"\$match\":\"*\"}}).include(\"repo\",\"path\",\"name\",\"@NIXL_SHA\")"
  # Only 200 is conclusive (no results is a valid 200); any other outcome skips
  # the release until the next cycle, so an Artifactory hiccup cannot fan out
  # spurious builds for every commit at once.
  http_code="$(curl -s --connect-timeout 10 --max-time 30 -o published.json -w '%{http_code}' \
    -u "${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}" -H 'Content-Type: text/plain' \
    --data-binary "${aql}" "${AQL_API_URL}")" || http_code=""
  if [ "${http_code}" != "200" ]; then
    echo "release/${ver}: AQL returned ${http_code:-<none>}, skipping this cycle"
    continue
  fi
  # include("@NIXL_SHA") narrows the response to that one property, so every
  # "value" in it is a marker sha.
  published="$(grep -oE '"value"[[:space:]]*:[[:space:]]*"[0-9a-f]{8}"' published.json \
    | grep -oE '[0-9a-f]{8}' || true)"

  base="$(git merge-base origin/main "origin/release/${ver}")"
  candidates="$(git rev-list --first-parent "${base}..origin/release/${ver}" | head -"${MAX_COMMITS}")"

  n_cand=0; n_build=0
  for sha in ${candidates}; do
    n_cand=$((n_cand+1))
    if printf '%s\n' "${published}" | grep -qx "${sha:0:8}"; then
      continue
    fi
    echo "${sha} ${ver}" >> triggers.txt
    n_build=$((n_build+1))
  done

  echo "release/${ver}: candidates=${n_cand} to_build=${n_build}"
done

echo "=== Poller summary: $(wc -l < triggers.txt) build(s) to trigger ==="
cat triggers.txt
