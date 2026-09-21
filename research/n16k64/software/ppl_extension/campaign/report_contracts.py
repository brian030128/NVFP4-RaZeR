"""Shared validation for canonicalized quality-report policy records."""
from __future__ import annotations


def validate_policy_contract(report, expected, context, mapping_field="evaluation"):
    """Validate policy membership and the sequence-bearing plan records.

    Quality reports are written by ``runtime.atomic_json`` with sorted mapping
    keys.  The order of the ``evaluation`` object is therefore serialization
    order, not frozen evaluation order.  The ``plan`` and ``installs`` arrays
    retain the authoritative order.
    """
    expected = tuple(expected)
    mapping_names = tuple(report[mapping_field])
    if len(mapping_names) != len(expected) or set(mapping_names) != set(expected):
        raise RuntimeError(f"{context} {mapping_field} policy membership drift")
    for field in ("plan", "installs"):
        observed = tuple(row["name"] for row in report[field])
        if observed != expected:
            raise RuntimeError(f"{context} {field} policy order drift")
