# Report template

## No use case

If no use case was selected or routing remains unresolved, output the matching
outcome line:

1. No route applies: `**Outcome:** No documented use case matched.`
2. Routing is unresolved: `**Outcome:** Routing incomplete — missing: <fact or
   choice>.`

When the request asks for investigation, evidence, or limitations, optionally
add at most one `**Evidence:**` line and one `**Limitation:**` line. Omit every
other section.

## Selected use cases

Use one report. Start with `**Outcome:**` and one sentence stating what is now
known or working. Include only sections with useful content.

## Findings

1. <Observed fact> — <decisive log, path, setting, version, or check>.

## Attribution

1. <NIXL | integrating software | host environment | unknown> — <first
   supported divergence>. Label any inference or hypothesis.

## Changes

1. <Applied change and location>. Revert: <how>.

## Verification

1. <Layer>: <check run on exact target> → <observed result>.

## Unresolved

1. <Blocked or untested layer, or issue left by a failed check> — <reason or
   missing evidence>. Do not repeat a result already listed under
   **Verification**.

## Next

1. <Single most useful remaining action and who owns it>.

## Rules

1. Omit empty sections; never write `None`, `N/A`, or placeholders.
2. Keep observations, attribution, changes, and verification separate.
3. List only applied changes. Put proposed changes under **Next**.
4. Verification includes only checks actually run; do not infer later layers.
5. Do not restate the request, narrate the process, dump logs, repeat facts,
   add generic advice, or end with another summary.
