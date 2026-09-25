#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../../..")
TRAP_ADD_SCRIPT="${REPO_ROOT}/contrib/trap_add.sh"

assert_eq() {
    local expected="$1"
    local actual="$2"
    local msg="$3"
    if [ "$expected" != "$actual" ]; then
        printf 'FAIL: %s\n  expected: %q\n  actual:   %q\n' "$msg" "$expected" "$actual" >&2
        exit 1
    fi
}

# Test 1: Single trap_add on EXIT executes on exit.
out=$(
    source "$TRAP_ADD_SCRIPT"
    trap_add 'echo "only_one"' EXIT
)
assert_eq "only_one" "$out" "Single trap_add should execute on EXIT"

# Test 2: Multiple trap_add calls append rather than replace existing EXIT traps.
out=$(
    source "$TRAP_ADD_SCRIPT"
    trap_add 'echo "first"' EXIT
    trap_add 'echo "second"' EXIT
    trap_add 'echo "third"' EXIT
)
expected=$(printf 'first\nsecond\nthird')
assert_eq "$expected" "$out" "Multiple trap_add calls should run all handlers in order"

# Test 3: trap_add preserves a prior plain `trap ... EXIT` in the same shell.
out=$(
    source "$TRAP_ADD_SCRIPT"
    trap 'echo "legacy"' EXIT
    trap_add 'echo "appended"' EXIT
)
expected=$(printf 'legacy\nappended')
assert_eq "$expected" "$out" "trap_add should preserve existing trap in same shell"

# Test 4: Subshell plain trap runs on subshell exit without affecting parent trap_add handlers.
out=$(
    source "$TRAP_ADD_SCRIPT"
    trap_add 'echo "parent_1"' EXIT
    trap_add 'echo "parent_2"' EXIT
    (
        trap 'echo "sub_cleanup"' EXIT
    )
    echo "between_sub_and_parent"
)
expected=$(printf 'sub_cleanup\nbetween_sub_and_parent\nparent_1\nparent_2')
assert_eq "$expected" "$out" "Subshell trap should run on subshell exit and preserve parent traps"

# Test 5: Different signals maintain separate trap lists without clobbering.
out=$(
    source "$TRAP_ADD_SCRIPT"
    trap_add 'echo "exit_1"' EXIT
    trap_add 'echo "usr1_1"' USR1
    trap_add 'echo "usr2_1"' USR2
    trap_add 'echo "exit_2"' EXIT
    trap_add 'echo "usr1_2"' SIGUSR1
    kill -USR1 "$BASHPID"
    kill -USR2 "$BASHPID"
)
expected=$(printf 'usr1_1\nusr1_2\nusr2_1\nexit_1\nexit_2')
assert_eq "$expected" "$out" "trap_add should maintain independent trap lists per signal"

# Test 6: Missing arguments or missing signal name returns non-zero.
if (
    source "$TRAP_ADD_SCRIPT"
    trap_add 2>/dev/null
); then
    echo "FAIL: trap_add without arguments should return non-zero" >&2
    exit 1
fi

if (
    source "$TRAP_ADD_SCRIPT"
    trap_add 'echo "missing_sig"' 2>/dev/null
); then
    echo "FAIL: trap_add without signal name should return non-zero" >&2
    exit 1
fi

echo "PASS: All trap_add tests succeeded."
