# Wheel inspection tool

`.ci/scripts/wheel_inspect` compares a candidate NIXL wheel set against a reference release and
reports what a release sign-off needs: the SPCX and DDN Infinia plugins are bundled, which SPCX
and UCX commits went in, which files were added, deleted or changed, the plugin sets, and the
wheel sizes. It carries no per-release expectations: whatever differs from the reference is
listed and a person judges it.

It is plain Python (3.10+, standard library only), so it runs in `CI_PYTHON_IMAGE` or on any
workstation. Nothing from a wheel is executed or loaded.

## Running it

```bash
# Two released versions from PyPI
python3 .ci/scripts/wheel_inspect release-run \
    --candidate pypi:1.5.2 --reference pypi:1.5.1 --out-dir /tmp/inspect

# A folder the nightly wheel build published, against the latest earlier PyPI release
ARTIFACTORY_USER=... ARTIFACTORY_TOKEN=... \
python3 .ci/scripts/wheel_inspect release-run \
    --candidate artifactory:https://artifactory.nvidia.com/artifactory/sw-nbu-swx-nixl-pypi-local/release/<ver>/<build> \
    --reference auto --out-dir /tmp/inspect --no-publish

# Two directories already on disk
python3 .ci/scripts/wheel_inspect inspect \
    --candidate /path/to/candidate --reference /path/to/reference --out-dir /tmp/inspect
```

`release-run` downloads both sides into `<out-dir>/candidate` and `<out-dir>/reference`
(verifying every sha256 against the index), inspects them, and, for an `artifactory:` candidate
without `--no-publish`, uploads the report into the candidate folder. `--reference auto` picks
the highest released version on PyPI below the candidate's; pre-releases are never chosen.
Wheels already downloaded are reused when their checksum still matches, so one `--out-dir` can
serve successive runs. `fetch`, `resolve-reference` and `publish-report` are the same steps one
at a time.

Each side must be a complete set: every CUDA x Python x architecture combination present, plus
exactly one meta wheel, all of one version. Anything else stops the run before a claim is made.
The two sides may cover different combinations (a new Python version, a dropped CUDA line); the
file comparison then runs on the shared ones and the rest is listed.

## The build manifest

The nightly wheel build publishes `build-manifest.json` next to the wheels (see
[`nixl-ci-build-wheel-nightly`](ci-overview.md#nixl-ci-build-wheel-nightly-standalone)). When a
side has one (found as `<dir>/build-manifest.json`, or passed with `--candidate-manifest` /
`--reference-manifest`), the tool first confirms it describes exactly these files, by name and
sha256, and that the build marked it complete. A manifest that fails this is not used: check 3
fails with the reason, and a manifest passed explicitly stops the run.

A usable manifest feeds checks 2 to 4 and the report's **Build provenance** section: NIXL
commit, UCX declared and measured, SPCX ref and commit, Infinia image, base image and torch list
per CUDA line, build tool versions, and per wheel which torch versions were built or skipped and
why. When both sides have one, every row is marked changed or unchanged. PyPI releases have no
manifest; for those the tool falls back to what the wheel carries (the `## Build info` block in
the README, and the UCX release string in `libucp`).

## The checks

| # | Check | Result |
| --- | --- | --- |
| 1 | SPCX plugin bundling | PASS when both SPCX UCX modules are in every binary wheel |
| 2 | DDN Infinia bundling | PASS when the plugin is in every binary wheel, links the DDN runtime and no `libred_*` library is bundled; FAIL also when the manifest says Infinia was not built |
| 3 | SPCX built from the expected ref | PASS with the manifest's ref and commit, which the wheel README must agree with; SKIPPED without a manifest; FAIL on an unusable manifest or a disagreement |
| 4 | UCX version and commit | PASS when every wheel carries the same UCX release and revision and they match the manifest |
| 5 | Files added / removed / changed | REVIEW lists per file with the combinations it applies to; PASS only when nothing differs |
| 6 | NIXL plugins bundled | PASS when the set is identical across the wheels of a CUDA line; additions and removals since the reference are listed |
| 7 | UCX plugins bundled | same rule for the UCX modules |
| 8 | Compressed wheel size | REVIEW: per wheel and total, with the members that moved by 1 MB or more |

`PASS` and `FAIL` appear only where code can judge, `REVIEW` marks a list for a person, and
`SKIPPED` names the missing input. Check 5 applies the rules the 1.4.1 audit learned the hard
way: the release version is stripped from both the `.dist-info/` and `.data/` prefixes; the
8-hex auditwheel suffix is stripped from vendored library names and a moved suffix is reported
as *rehashed*, not as an add plus a remove; and "did NIXL code change" is answered from the
unrelocated `.nixl_cuXX.mesonpy.libs/` copies, which auditwheel never touches. A large
`.nv_fatbin` size change with identical device code is a compression-mode difference; the tool
compares `cuobjdump --list-elf` output when `cuobjdump` is available (`--cuobjdump`, else
`PATH`) and says so when it is not.

## Output

`wheel-inspection.json`, `.md` and `.html` in `--out-dir`, plus the results table on stdout. The
JSON `summary` counts the states; the Markdown and HTML are rendered from the same document.

| Exit code | Meaning |
| --- | --- |
| 0 | the report was produced, whatever the checks say; read `summary` for `FAIL` |
| 1 | usage error |
| 2 | no trustworthy report: incomplete set, failed download, unreadable wheel, bad explicit manifest |

Downloads are limited to `pypi.org`, `files.pythonhosted.org` and the two NVIDIA Artifactory
hosts; the Artifactory credentials are read from `ARTIFACTORY_USER` / `ARTIFACTORY_TOKEN` and
sent only there.
