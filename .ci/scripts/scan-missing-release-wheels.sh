#!/bin/bash -eE
# Scan release/* branches for commits with no published wheel set, and write
# triggers.txt (one "<sha> <ver>" line per missing build) for the poller's
# trigger step.
#
# A commit counts as published when a build folder under release/<ver>/ carries
# it as a NIXL_SHA property, which the wheel job sets only on a fully green
# build. Commits with a fresh .inflight/<sha8> reservation are already building.
#
# Env (set by build-wheel-release-poller-matrix.yaml):
#   MIN_RELEASE   - releases older than this are not built
#   MAX_COMMITS   - newest N first-parent commits to check per release branch
#   RESERVE_TTL_MIN - minutes an .inflight reservation counts as live
#   NIXL_REPO_URL, AQL_API_URL, WHEEL_REPO_NAME, WHEEL_REPO_URL, WHEEL_REPO_API,
#   ARTIFACTORY_USER, ARTIFACTORY_TOKEN

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

  # repo/path/name are mandatory in any items .include() - Artifactory rejects
  # the query without them ("for permissions reasons").
  aql="items.find({\"repo\":\"${WHEEL_REPO_NAME}\",\"type\":\"folder\",\"path\":\"release/${ver}\",\"@NIXL_SHA\":{\"\$match\":\"*\"}}).include(\"repo\",\"path\",\"name\",\"@NIXL_SHA\")"
  # Only 200 is conclusive (no results is a valid 200); anything else skips the
  # release so an Artifactory hiccup cannot fan out a build for every commit.
  http_code="$(curl -s --connect-timeout 10 --max-time 30 -o published.json -w '%{http_code}' \
    -u "${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}" -H 'Content-Type: text/plain' \
    --data-binary "${aql}" "${AQL_API_URL}")" || http_code=""
  if [ "${http_code}" != "200" ]; then
    echo "release/${ver}: published AQL returned ${http_code:-<none>}, skipping this cycle"
    head -c 500 published.json; echo
    continue
  fi
  published="$(grep -oE '"value"[[:space:]]*:[[:space:]]*"[0-9a-f]{8}"' published.json \
    | grep -oE '[0-9a-f]{8}' || true)"

  # Reservations, aged here rather than in the query: AQL rejected both relative
  # date operators tried against this instance, and ?list already returns
  # lastModified per file. 404 just means nothing has been reserved yet.
  http_code="$(curl -s --connect-timeout 10 --max-time 30 -o inflight.json -w '%{http_code}' \
    -u "${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}" \
    "${WHEEL_REPO_API}/release/${ver}/.inflight/?list&listFolders=0")" || http_code=""
  if [ "${http_code}" != "200" ] && [ "${http_code}" != "404" ]; then
    echo "release/${ver}: in-flight listing returned ${http_code:-<none>}, skipping this cycle"
    head -c 500 inflight.json; echo
    continue
  fi
  reserved=""
  if [ "${http_code}" = "200" ]; then
    now_epoch="$(date -u +%s)"
    # One record per line. tr drops the trailing newline, hence the `|| [ -n ]`
    # guard - without it the last reservation is silently ignored.
    reserved="$(tr -d '\n' < inflight.json | sed 's/{/\n{/g' \
      | while IFS= read -r rec || [ -n "${rec}" ]; do
          name="$(printf '%s' "${rec}" | grep -oE '"uri"[[:space:]]*:[[:space:]]*"/[0-9a-f]{8}"' | grep -oE '[0-9a-f]{8}')"
          ts="$(printf '%s' "${rec}" | grep -oE '"lastModified"[[:space:]]*:[[:space:]]*"[^"]+"' | sed 's/.*"\([^"]*\)"$/\1/')"
          if [ -z "${name}" ] || [ -z "${ts}" ]; then continue; fi
          ts_epoch="$(date -u -d "${ts}" +%s 2>/dev/null)" || continue
          [ $(( (now_epoch - ts_epoch) / 60 )) -lt "${RESERVE_TTL_MIN}" ] && printf '%s\n' "${name}"
        done)"
  fi

  base="$(git merge-base origin/main "origin/release/${ver}")"
  candidates="$(git rev-list --first-parent "${base}..origin/release/${ver}" | head -"${MAX_COMMITS}")"

  n_cand=0; n_build=0; n_flight=0
  for sha in ${candidates}; do
    n_cand=$((n_cand+1))
    if printf '%s\n' "${published}" | grep -qx "${sha:0:8}"; then
      continue
    fi
    if printf '%s\n' "${reserved}" | grep -qx "${sha:0:8}"; then
      echo "release/${ver}: ${sha:0:8} already has a build in flight"
      n_flight=$((n_flight+1))
      continue
    fi
    # Reserve before triggering; a failed reservation only risks a duplicate.
    curl -fsS --connect-timeout 10 --max-time 30 -o /dev/null \
      -u "${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}" -X PUT --data-binary '' \
      "${WHEEL_REPO_URL}/release/${ver}/.inflight/${sha:0:8}" \
      || echo "release/${ver}: could not reserve ${sha:0:8}, a duplicate build is possible"
    echo "${sha} ${ver}" >> triggers.txt
    n_build=$((n_build+1))
  done

  echo "release/${ver}: candidates=${n_cand} in_flight=${n_flight} to_build=${n_build}"
done

echo "=== Poller summary: $(wc -l < triggers.txt) build(s) to trigger ==="
cat triggers.txt
