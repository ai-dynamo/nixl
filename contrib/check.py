#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check a disposable snapshot; never format or install into the contributor tree."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

IMAGE = "nixl-local-checks:ubuntu24"
DEFAULT_TESTS = ["posix_plugin_test", "posix_aio_backend_test"]

DOCKERFILE = r"""
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates git build-essential ccache cmake pkg-config ninja-build clang-format-19 \
    python3 python3-dev python3-venv libaio-dev liburing-dev libgtest-dev libgmock-dev \
    curl nodejs npm && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/checks && /opt/checks/bin/pip install --no-cache-dir \
    pre-commit==4.2.0 meson==1.7.2 pybind11==2.13.6 meson-python==0.18.0 \
    setuptools wheel
ENV PATH="/opt/checks/bin:/usr/lib/ccache:${PATH}"
# Match .github/workflows/fern-check.yml (Node 22 and Fern 5.90.1).
RUN npm install -g n@10.2.0 && n 22 && npm install -g fern-api@5.90.1
# Ubuntu supplies gcc/g++ wrappers but omits the generic compiler aliases.
RUN ln -s /usr/bin/ccache /usr/lib/ccache/cc && \
    ln -s /usr/bin/ccache /usr/lib/ccache/c++
ENV NIXL_CI_NON_GPU=1
ENV CCACHE_DIR=/cache/ccache
"""

LOADER_SOURCE = r"""
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "nixl.h"

int
main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    nixlAgent agent("local-plugin-check", nixlAgentConfig());
    nixl_mem_list_t mems;
    nixl_b_params_t params;
    return agent.getPluginParams(argv[1], mems, params) == NIXL_SUCCESS ? 0 : 1;
}
"""


def output(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def git_files(root):
    return subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    ).split(b"\0")


def snapshot(root, dest, ref=None):
    # A real .git directory works in Docker even when root is a linked worktree.
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(root), str(dest)],
        check=True,
    )
    if ref:
        subprocess.run(
            ["git", "checkout", "--quiet", "--detach", ref], cwd=dest, check=True
        )
        return
    for raw in set(git_files(dest)) | set(git_files(root)):
        if not raw:
            continue
        name = os.fsdecode(raw)
        source, target = root / name, dest / name
        if target.is_symlink() or target.is_file():
            target.unlink()
        if source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.readlink(source))
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        elif source.is_dir():
            raise RuntimeError(
                f"Submodule/directory {name}: validate in its own checkout"
            )
    subprocess.run(["git", "add", "-A"], cwd=dest, check=True)
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="Local checks",
        GIT_AUTHOR_EMAIL="checks@localhost",
        GIT_COMMITTER_NAME="Local checks",
        GIT_COMMITTER_EMAIL="checks@localhost",
    )
    subprocess.run(
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "Local validation snapshot",
        ],
        cwd=dest,
        env=env,
        check=True,
    )


class Checks:
    def __init__(self, report):
        self.report = report
        self.results = []

    def record(self, name, status, detail=""):
        self.results.append(dict(check=name, status=status, detail=detail))
        print(f"{status:10} {name}: {detail}", flush=True)
        return status == "PASS"

    def run(self, name, args, *, cwd=None, env=None, stdin=None, empty=False):
        logfile = self.report / (name + ".log")
        print(f"Running {name} ...", flush=True)
        try:
            with logfile.open("w") as log:
                log.write(shlex.join(map(str, args)) + "\n")
                log.flush()
                result = subprocess.run(
                    args,
                    cwd=cwd,
                    env=env,
                    input=stdin,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            command_output = logfile.read_text().partition("\n")[2]
            ok = result.returncode == 0 and (not empty or not command_output.strip())
            if not ok:
                print("\n".join(command_output.splitlines()[-20:]), flush=True)
            return self.record(name, "PASS" if ok else "FAIL", logfile.name)
        except FileNotFoundError as exc:
            return self.record(name, "INCOMPLETE", str(exc))

    def lint(self, base):
        files = (
            subprocess.check_output(
                ["git", "diff", "--name-only", "-z", "--diff-filter=ACMR", base, "HEAD"]
            )
            .decode()
            .split("\0")
        )
        files = [f for f in files if f]
        if files:
            self.run(
                "pre-commit",
                ["pre-commit", "run", "--show-diff-on-failure", "--files", *files],
            )
            patch = subprocess.check_output(["git", "diff", "--binary"])
            (self.report / "formatting.patch").write_bytes(patch)
            # Later checks must see the submitted snapshot, not formatter edits.
            subprocess.run(
                ["git", "reset", "--hard", "HEAD"],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        else:
            self.record("pre-commit", "PASS", "no changed files")
        cpp = [
            f
            for f in files
            if re.search(r"\.(cpp|h|cc|c|cxx|hpp|cu|cuh)$", f)
            and not f.startswith("examples/device/ep/")
        ]
        if cpp:
            diff = output(
                "git",
                "diff",
                "--no-ext-diff",
                "--no-color",
                "-U0",
                base,
                "HEAD",
                "--",
                *cpp,
            )
            self.run(
                "clang-format",
                [
                    "clang-format-diff-19",
                    "-p1",
                    "-style=file",
                    "-binary",
                    "clang-format-19",
                    "-iregex",
                    r".*\.(cpp|h|cc|c|cxx|hpp|cu|cuh)$",
                ],
                stdin=diff + "\n",
                empty=True,
            )
        else:
            self.record("clang-format", "PASS", "no changed C/C++ files")
        self.run("copyright", ["bash", ".github/workflows/copyright-check.sh", base])
        for directory in ["src", "test", "benchmark", "examples/python"]:
            self.run(
                "prints-" + directory.replace("/", "-"),
                ["bash", ".ci/scripts/check_prints.sh", directory],
            )
        stats = output(
            "git",
            "diff",
            "--numstat",
            "--diff-filter=M",
            base,
            "HEAD",
            "--",
            ".",
            ":(exclude)subprojects/*",
        )
        added = sum(
            int(line.split("\t")[0])
            for line in stats.splitlines()
            if line.split("\t")[0].isdigit()
        )
        self.record(
            "pr-size",
            "PASS" if added <= 500 else "FAIL",
            f"{added}/500 added lines in modified files",
        )
        self.run("fern", ["fern", "check"], cwd="fern")
        self.run("fern-markdown", ["fern", "docs", "md", "check"], cwd="fern")

    def backend(self, args):
        build = Path("build-local").resolve()
        setup = [
            "meson",
            "setup",
            str(build),
            *args.meson_arg,
            "--buildtype=debug",
            "-Dwerror=true",
            "-Dbuild_tests=true",
            "-Dbuild_examples=true",
            "-Dstatic_plugins=",
            "-Ddisable_plugins=",
            f"-Denable_plugins={args.plugin}",
        ]
        if not self.run("configure", setup):
            return
        targets = json.loads(output("meson", "introspect", str(build), "--targets"))
        libraries = [
            Path(f)
            for t in targets
            for f in t["filename"]
            if Path(f).name == f"libplugin_{args.plugin}.so"
        ]
        if len(libraries) != 1:
            self.record(
                "backend-built",
                "FAIL",
                "selected backend has no unique shared library target",
            )
            return
        tests = json.loads(output("meson", "introspect", str(build), "--tests"))
        available = {t["name"] for t in tests}
        missing = set(args.test) - available
        if missing:
            self.record(
                "backend-tests",
                "FAIL",
                f"tests not registered: {sorted(missing)}; available: {sorted(available)}",
            )
            return
        if not self.run(
            "compile", ["meson", "compile", "-C", str(build), "-j", str(args.jobs)]
        ):
            return
        library = libraries[0]
        env = dict(os.environ, NIXL_PLUGIN_DIR=str(library.parent))
        # An existing host installation must not shadow the libraries just built.
        libdirs = dict.fromkeys(
            str(Path(f).parent)
            for t in targets
            for f in t["filename"]
            if ".so" in Path(f).name
        )
        env["LD_LIBRARY_PATH"] = ":".join(libdirs)
        if os.environ.get("LD_LIBRARY_PATH"):
            env["LD_LIBRARY_PATH"] += ":" + os.environ["LD_LIBRARY_PATH"]
        # Link to the core just built, and ask its loader to load this exact plugin.
        cores = [Path(f) for t in targets if t["name"] == "nixl" for f in t["filename"]]
        smoke = build / "load-plugin"
        if not cores:
            self.record("load-plugin", "FAIL", "NIXL core target missing")
            return
        core = cores[0]
        probe = build / "load-plugin.cpp"
        probe.write_text(LOADER_SOURCE)
        if self.run(
            "compile-loader",
            [
                "c++",
                "-std=c++20",
                "-I",
                "src/api/cpp",
                str(probe),
                str(core),
                "-Wl,-rpath," + str(core.parent),
                "-o",
                str(smoke),
            ],
        ):
            self.run("load-plugin", [str(smoke), args.plugin], env=env)
        self.run(
            "backend-tests",
            [
                "meson",
                "test",
                "-C",
                str(build),
                "--no-rebuild",
                "--print-errorlogs",
                "--num-processes",
                "1",
                *args.test,
            ],
            env=env,
        )
        # Meson returns success for skipped tests. Do not call that validated.
        testlog = build / "meson-logs/testlog.json"
        if testlog.exists():
            rows = [
                json.loads(line) for line in testlog.read_text().splitlines() if line
            ]
            shutil.copy2(testlog, self.report / "tests.jsonl")
            bad = [row["name"] for row in rows if row["result"] != "OK"]
            if bad or len(rows) < len(set(args.test)):
                self.record(
                    "test-coverage",
                    "INCOMPLETE",
                    f"missing, skipped or non-passing tests: {bad}",
                )
        else:
            self.record("test-coverage", "INCOMPLETE", "Meson test result log missing")
        logs = build / "meson-logs"
        shutil.copytree(logs, self.report / "meson-logs", dirs_exist_ok=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plugin",
        default="POSIX",
        help="backend to build and test (default: POSIX CPU smoke tests)",
    )
    p.add_argument(
        "--test",
        action="append",
        default=[],
        help="exact Meson test name; repeat for multiple tests",
    )
    p.add_argument(
        "--ref", help="check an exact commit or branch instead of local edits"
    )
    p.add_argument(
        "--lint-only", action="store_true", help="omit build and integration tests"
    )
    p.add_argument(
        "--base", help="PR target ref (default: upstream/main, origin/main, main)"
    )
    p.add_argument(
        "--host", action="store_true", help="use installed tools/SDKs instead of Docker"
    )
    p.add_argument(
        "--image",
        default=IMAGE,
        help="Docker image with tools and backend SDK; default image builds automatically",
    )
    p.add_argument(
        "--meson-arg",
        action="append",
        default=[],
        help="extra setup option, e.g. --meson-arg=-Ducx_path=/opt/ucx",
    )
    p.add_argument("--jobs", type=int, default=4, help="compile workers (default: 4)")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--report", help=argparse.SUPPRESS)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.plugin and not re.fullmatch(r"[A-Z][A-Z0-9_]*", args.plugin):
        p.error("--plugin must be an uppercase backend name")
    if not args.test:
        if args.plugin == "POSIX":
            args.test = DEFAULT_TESTS.copy()
        elif not args.lint_only:
            p.error("a custom --plugin requires at least one --test NAME")
    if args.jobs < 1:
        p.error("--jobs must be positive")
    if args.worker:
        if not Path(".git/nixl-check-snapshot").is_file():
            p.error("internal worker requires a disposable validation snapshot")
        report = Path(args.report)
        checks = Checks(report)
        try:
            checks.lint(args.base)
            if not args.lint_only:
                checks.backend(args)
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            checks.record("runner", "INCOMPLETE", str(exc))
        (report / "results.json").write_text(
            json.dumps(checks.results, indent=2) + "\n"
        )
        print(
            "\nScope: local lint"
            + (f", {args.plugin} build/load/tests" if not args.lint_only else "")
            + ". GPU, full CI matrix and packaging are not certified."
        )
        if any(r["status"] == "FAIL" for r in checks.results):
            return 1
        return 2 if any(r["status"] == "INCOMPLETE" for r in checks.results) else 0

    root = Path(output("git", "rev-parse", "--show-toplevel"))
    os.chdir(root)
    if output("git", "rev-parse", "--is-shallow-repository") == "true":
        p.error("full history is needed for copyright checks: git fetch --unshallow")
    base = args.base
    if not base:
        for candidate in ["upstream/main", "origin/main", "main"]:
            if (
                subprocess.run(
                    ["git", "rev-parse", "--verify", candidate],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            ):
                base = candidate
                break
    if not base:
        p.error("no target branch found; pass --base REF")
    target = output("git", "rev-parse", "--verify", (args.ref or "HEAD") + "^{commit}")
    base_sha = output("git", "merge-base", base, target)
    report_root = root / ".cache/local-checks"
    report_root.mkdir(parents=True, exist_ok=True)
    report = Path(tempfile.mkdtemp(prefix="run-", dir=report_root))
    cache = report_root / "tool-cache"
    cache.mkdir(exist_ok=True)
    print(
        f"Checking {args.ref or 'working tree'} against {base} ({base_sha[:12]})\nReport: {report}",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="nixl-checks-") as tmp:
        dest = Path(tmp) / "source"
        snapshot(root, dest, target if args.ref else None)
        (dest / ".git/nixl-check-snapshot").touch()
        (report / "source.json").write_text(
            json.dumps(
                dict(
                    head=target,
                    ref=args.ref or "working tree",
                    base=base_sha,
                    tree=output("git", "rev-parse", "HEAD^{tree}", cwd=dest),
                    plugin=args.plugin,
                    tests=args.test,
                    image=None if args.host else args.image,
                    command=sys.argv,
                ),
                indent=2,
            )
            + "\n"
        )
        worker = [
            "python3",
            str(Path(__file__).resolve()) if args.host else "/check.py",
            "--worker",
            "--base",
            base_sha,
            "--report",
            str(report) if args.host else "/report",
            "--jobs",
            str(args.jobs),
        ]
        worker += ["--plugin", args.plugin]
        if args.lint_only:
            worker += ["--lint-only"]
        for test in args.test:
            worker += ["--test", test]
        worker += ["--meson-arg=" + opt for opt in args.meson_arg]
        if args.host:
            rc = subprocess.run(worker, cwd=dest).returncode
        else:
            if args.image == IMAGE:
                context = Path(tmp) / "image"
                context.mkdir()
                (context / "Dockerfile").write_text(DOCKERFILE.lstrip())
                subprocess.run(
                    ["docker", "build", "-t", IMAGE, str(context)], check=True
                )
            rc = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--ulimit",
                    "nofile=65536:65536",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "-e",
                    "HOME=/cache",
                    "-v",
                    f"{cache}:/cache",
                    "-v",
                    f"{Path(__file__).resolve()}:/check.py:ro",
                    "-v",
                    f"{dest}:/src",
                    "-v",
                    f"{report}:/report",
                    "-w",
                    "/src",
                    args.image,
                    *worker,
                ]
            ).returncode
    print(
        f"\nReport: {report}\nExit {rc}: 0=passed selected scope, 1=failed, 2=incomplete"
    )
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f"INCOMPLETE: {error}", file=sys.stderr)
        sys.exit(2)
