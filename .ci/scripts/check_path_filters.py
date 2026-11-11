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
"""Check that every given file is registered in .ci/path-filters.yaml.

Reference implementation of the path-filter match rules; the Jenkins
dispatcher mirrors them in Groovy. See .ci/docs/ci-overview.md, "Path
filtering".
"""

from __future__ import annotations

import argparse
import re
import sys

import yaml

DEFAULT_CONFIG = ".ci/path-filters.yaml"
NAME_RE = re.compile(r"[A-Za-z0-9_]+")
FORBIDDEN_RE = re.compile(r"[?\[\]!\\\s]")


class ConfigError(Exception):
    pass


class _StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ConfigError(f"duplicate key '{key}'")
            seen.add(key)
        return super().construct_mapping(node, deep)


def _segment_regex(segment: str) -> str:
    if segment == "**":
        return ".*"
    return "".join("[^/]*" if c == "*" else re.escape(c) for c in segment)


def compile_entry(entry: str) -> tuple[str, int]:
    """Return (regex, specificity) for one path entry."""
    if not entry:
        raise ConfigError("empty entry")
    if FORBIDDEN_RE.search(entry):
        raise ConfigError("only '*' and '**' wildcards are supported")
    anchored = entry.startswith("/")
    text = entry[1:] if anchored else entry
    text = text[:-1] if text.endswith("/") else text
    if not text:
        raise ConfigError("empty entry")
    segments = text.split("/")
    for i, seg in enumerate(segments):
        if not seg:
            raise ConfigError("empty path segment")
        if "**" in seg and (seg != "**" or i != len(segments) - 1):
            raise ConfigError("'**' is only allowed as the whole last segment")
    body = "/".join(_segment_regex(s) for s in segments)
    if len(segments) == 1 and not anchored:
        return f"(?:.*/)?{body}(?:/.*)?", 0
    if "*" not in text:
        body += "(?:/.*)?"
    star = text.find("*")
    return body, len(text) if star < 0 else star


def load_components(path: str) -> dict[str, list[tuple[str, int]]]:
    """Parse and validate the config; return compiled entries per component."""
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.load(f, Loader=_StrictLoader)
    except OSError as e:
        raise ConfigError(e.strerror or str(e)) from e
    except yaml.YAMLError as e:
        raise ConfigError("invalid YAML: " + "; ".join(str(e).splitlines())) from e
    if not isinstance(data, dict):
        raise ConfigError(
            "top-level must be a mapping with the single key 'components'"
        )
    for key in data:
        if key != "components":
            raise ConfigError(
                f"unexpected top-level key '{key}'; only 'components' is allowed"
            )
    components = data.get("components")
    if not isinstance(components, dict):
        raise ConfigError(
            "'components' must be a mapping of component name to list of entries"
        )
    compiled: dict[str, list[tuple[str, int]]] = {}
    for name, entries in components.items():
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise ConfigError(f"invalid component name '{name}'")
        if not isinstance(entries, list):
            raise ConfigError(f"component '{name}' must be a list of path entries")
        compiled[name] = []
        for entry in entries:
            if not isinstance(entry, str):
                raise ConfigError(
                    f"component '{name}': entry {entry!r} is not a string"
                )
            try:
                compiled[name].append(compile_entry(entry))
            except ConfigError as e:
                raise ConfigError(
                    f"component '{name}': invalid entry '{entry}': {e}"
                ) from e
    if "all" not in compiled:
        raise ConfigError("component 'all' is missing")
    return compiled


def components_for(path: str, compiled: dict[str, list[tuple[str, int]]]) -> set[str]:
    """Components of the most specific entries matching path; empty if none."""
    best = -1
    result: set[str] = set()
    for name, entries in compiled.items():
        for regex, specificity in entries:
            if specificity < best or not re.fullmatch(regex, path):
                continue
            if specificity > best:
                best = specificity
                result = set()
            result.add(name)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(argv)
    try:
        compiled = load_components(args.config)
    except ConfigError as e:
        print(f"{args.config}: {e}", file=sys.stderr)
        return 2
    rc = 0
    for path in args.files:
        if not components_for(path, compiled):
            print(
                f"{path}: not registered in {args.config}; add it to a component or to 'all'"
            )
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
