# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Summarize named diagnostic timestamps without duplicating the native phase schema."""


def summarize_profile(profile: dict) -> dict:
    """Return first/last observed offsets in microseconds from each operation's earliest entry, and writer counts.

    These are phase boundaries across different warps, not additive phase durations. Missing writers are ignored.
    """
    bounds = {}
    for name, timestamps in profile.items():
        values = [value for row in timestamps.tolist() for value in row if value != 0]
        if values:
            bounds[name] = min(values), max(values), len(values)
    result = {}
    for name, (first, last, writers) in bounds.items():
        entry = name.split(".", 1)[0] + ".entry"
        if entry not in bounds:
            raise ValueError(
                f"{name} has samples without {entry}; clear the profile, then record one pair or replay"
            )
        origin = bounds[entry][0]
        result[name] = dict(
            first_us=(first - origin) / 1000,
            last_us=(last - origin) / 1000,
            writers=writers,
        )
    return result
