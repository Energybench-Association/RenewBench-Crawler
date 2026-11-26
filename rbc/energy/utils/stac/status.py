"""STAC STATUS FIELD.

The STAC energy extension's energy:status definition via `StatusDefiner`.
"""

from rbc.energy.utils.stac.base import FieldDefiner

# Set: energy:status values STAC Items may be defined as
STATUSES = frozenset(
    {
        "active",
        "inactive",
        "under_construction",
        "decommissioned",
    }
)


class StatusDefiner(FieldDefiner):
    """Define an operator's EGEs' energy:status from its own values."""

    FIELD = "status"
    VALUES = STATUSES
