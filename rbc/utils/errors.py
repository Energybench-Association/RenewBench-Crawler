"""ERRORS.

The rbc package's own exceptions, shared by its downloaders, processors and helpers.
"""


class MissingDataError(Exception):
    """Raised when a data source is missing data that a task is trying to download."""

    pass


class DataStructureError(Exception):
    """Raised when a data source changes its structure significantly."""

    pass


class RateLimitError(Exception):
    """Raised when the API rate limit has been reached and downloading must be stopped."""

    pass


class InvalidError(Exception):
    """Raised when provided values/arguments are invalid in some way that force a stop."""

    pass
