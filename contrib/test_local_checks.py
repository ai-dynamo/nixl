# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner regressions: python3 -m unittest discover -s contrib -p 'test_local_checks.py'."""

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "local_checks", Path(__file__).with_name("check.py")
)
assert SPEC is not None and SPEC.loader is not None
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


class SnapshotTests(unittest.TestCase):
    def test_dirty_linked_worktree_preserves_index_and_captures_deletions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()

            def git(*args, cwd=root):
                return subprocess.check_output(
                    ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL
                )

            git("init", "-b", "main")
            git("config", "commit.gpgsign", "false")
            git("config", "core.hooksPath", "/dev/null")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.com")
            for name in ["changed", "staged-delete", "unstaged-delete", "rename-me"]:
                (root / name).write_text("original\n")
            (root / ".gitignore").write_text("ignored\n")
            git("add", ".")
            git("commit", "-qm", "Initial")
            worktree = Path(tmp) / "worktree"
            git("worktree", "add", "-b", "topic", str(worktree))
            (worktree / "changed").write_text("staged\n")
            git("add", "changed", cwd=worktree)
            (worktree / "changed").write_text("unstaged\n")
            git("rm", "staged-delete", cwd=worktree)
            (worktree / "unstaged-delete").unlink()
            git("mv", "rename-me", "renamed", cwd=worktree)
            (worktree / "new file.py").write_text("new\n")
            (worktree / "ignored").write_text("scratch\n")
            before = git("diff", "--cached", "--binary", cwd=worktree)
            status = git("status", "--porcelain", cwd=worktree)
            dest = Path(tmp) / "snapshot"
            checks.snapshot(worktree, dest)
            self.assertEqual((dest / "changed").read_text(), "unstaged\n")
            self.assertTrue((dest / "new file.py").exists())
            self.assertTrue((dest / "renamed").exists())
            for name in ["staged-delete", "unstaged-delete", "rename-me", "ignored"]:
                self.assertFalse((dest / name).exists(), name)
            self.assertTrue((dest / ".git").is_dir())
            self.assertEqual(before, git("diff", "--cached", "--binary", cwd=worktree))
            self.assertEqual(status, git("status", "--porcelain", cwd=worktree))
            # An explicit ref must ignore every local edit, without switching branches.
            committed = Path(tmp) / "committed"
            checks.snapshot(
                worktree,
                committed,
                git("rev-parse", "HEAD", cwd=worktree).decode().strip(),
            )
            self.assertEqual((committed / "changed").read_text(), "original\n")
            self.assertTrue((committed / "staged-delete").exists())
            self.assertTrue((committed / "rename-me").exists())
            self.assertFalse((committed / "new file.py").exists())
            self.assertEqual(status, git("status", "--porcelain", cwd=worktree))
            self.assertEqual(
                git("branch", "--show-current", cwd=worktree).strip(), b"topic"
            )


class ResultTests(unittest.TestCase):
    def test_copyright_checks_earlier_branch_changes_in_both_snapshot_modes(self):
        copyright_script = (
            Path(__file__).resolve().parents[1] / ".github/workflows/copyright-check.sh"
        ).read_text()
        year = datetime.now(timezone.utc).year
        header = (
            f"# SPDX-FileCopyrightText: Copyright (c) {year} "
            "NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()

            def git(*args):
                return (
                    subprocess.check_output(
                        ["git", *args], cwd=root, stderr=subprocess.DEVNULL
                    )
                    .decode()
                    .strip()
                )

            git("init", "-b", "main")
            git("config", "commit.gpgsign", "false")
            git("config", "core.hooksPath", "/dev/null")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.com")
            script = root / ".github/workflows/copyright-check.sh"
            script.parent.mkdir(parents=True)
            script.write_text(copyright_script)
            git("add", ".")
            git("commit", "-qm", "Base")
            base = git("rev-parse", "HEAD")
            (root / "early.py").write_text("# Missing SPDX header\n")
            git("add", ".")
            git("commit", "-qm", "Earlier branch change")
            (root / "latest.py").write_text(header)
            git("add", ".")
            git("commit", "-qm", "Latest branch change")
            ref = git("rev-parse", "HEAD")
            (root / "uncommitted.py").write_text(header)

            for mode, target in [("working-tree", None), ("ref", ref)]:
                with self.subTest(mode=mode):
                    dest = Path(tmp) / mode
                    checks.snapshot(root, dest, target)
                    report = Path(tmp) / (mode + "-report")
                    report.mkdir()
                    runner = checks.Checks(report)
                    real_run = runner.run

                    def copyright_only(name, command, **kwargs):
                        if name == "copyright":
                            return real_run(name, command, **kwargs)
                        return True

                    old = os.getcwd()
                    try:
                        os.chdir(dest)
                        with patch.object(runner, "run", side_effect=copyright_only):
                            runner.lint(base)
                    finally:
                        os.chdir(old)
                    result = next(
                        r for r in runner.results if r["check"] == "copyright"
                    )
                    self.assertEqual(result["status"], "FAIL")
                    self.assertIn("early.py", (report / "copyright.log").read_text())

    def test_missing_tool_and_silent_failure_never_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = checks.Checks(Path(tmp))
            self.assertFalse(runner.run("missing", ["/nonexistent-nixl-tool"]))
            self.assertEqual(runner.results[-1]["status"], "INCOMPLETE")
            self.assertFalse(runner.run("silent", ["sh", "-c", "exit 7"], empty=True))
            self.assertEqual(runner.results[-1]["status"], "FAIL")

    def test_skipped_backend_test_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.getcwd()
            os.chdir(tmp)
            self.addCleanup(os.chdir, old)
            report = Path(tmp) / "report"
            report.mkdir()
            logs = Path("build-local/meson-logs")
            logs.mkdir(parents=True)
            (logs / "testlog.json").write_text(
                json.dumps(dict(name="real-test", result="SKIP")) + "\n"
            )
            runner = checks.Checks(report)
            args = checks.parser().parse_args(
                ["--plugin", "FAKE", "--test", "real-test"]
            )
            targets = [
                dict(name="FAKE", filename=[str(Path("libplugin_FAKE.so").resolve())]),
                dict(name="nixl", filename=[str(Path("libnixl.so").resolve())]),
            ]
            with (
                patch.object(
                    checks,
                    "output",
                    side_effect=[
                        json.dumps(targets),
                        json.dumps([dict(name="real-test")]),
                    ],
                ),
                patch.object(runner, "run", return_value=True),
            ):
                runner.backend(args)
            self.assertEqual(runner.results[-1]["status"], "INCOMPLETE")

    def test_backend_cannot_silently_disappear(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = checks.Checks(Path(tmp))
            args = checks.parser().parse_args(
                ["--plugin", "FAKE", "--test", "real-test"]
            )
            with (
                patch.object(checks, "output", return_value="[]"),
                patch.object(runner, "run", return_value=True),
            ):
                runner.backend(args)
            self.assertEqual(runner.results[-1]["status"], "FAIL")

    def test_internal_worker_rejects_normal_checkout(self):
        script = Path(__file__).with_name("check.py").resolve()
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["python3", str(script), "--worker", "--report", tmp],
                cwd=tmp,
                timeout=5,
                capture_output=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"disposable validation snapshot", result.stderr)

    def test_missing_option_value_terminates(self):
        script = Path(__file__).with_name("check.py")
        result = subprocess.run(
            ["python3", str(script), "--plugin"], timeout=5, capture_output=True
        )
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
