"""STAC MEASUREMENT TYPE FIELD.

The STAC energy extension's energy:measurement_type definition via `MeasurementTypeDefiner`.
"""

from rbc.energy.utils.stac.base import FieldDefiner

# Set: energy:measurement_type values STAC Items may be defined as
MEASUREMENT_TYPES = frozenset(
    {
        "gross_output",
        "net_output",
        "simulated_output",
        "estimated_output",
        "forecast_output",
        "calculated_output",
        "curtailed_output",
        "unknown",
    }
)


class MeasurementTypeDefiner(FieldDefiner):
    """Define an operator's EGEs' energy:measurement_type from its own values/mappings."""

    FIELD = "measurement_type"
    VALUES = MEASUREMENT_TYPES
