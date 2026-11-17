# Debug Session Pitfalls

Use this reference when an investigation is expanding, losing provenance, or
turning into a fix plan before the failing layer is proven.

## Common Pitfalls

- Starting root-cause analysis before there is a failing command, log excerpt,
  issue link, or reproduction clue.
- Treating host evidence as container or pod evidence when the workload fails
  inside a different runtime boundary.
- Copying broad raw logs, secrets, private hostnames, or internal IPs into the
  worklog instead of concise redacted excerpts.
- Mutating repo files, services, clusters, drivers, images, or manifests under a
  stale approval after new evidence changes the plan.
- Debugging transfer logic when importability, plugin discovery, or connector
  setup has already failed.
- Treating latest public docs as source truth for an older installed framework,
  NIXL package, or container image.
- Turning a debug worklog into a broad refactor or performance-tuning project.

## Recovery Moves

- Re-route to install/importability readiness if the failure occurs before NIXL
  or the connector is usable.
- Keep each worklog entry tied to source IDs, timestamps, command output, and
  unresolved `TBD-*` items.
- When evidence changes the hypothesis, restate the exact next mutation and get
  explicit approval before acting.
