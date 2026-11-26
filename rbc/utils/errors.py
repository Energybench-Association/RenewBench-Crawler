"""ERRORS.

The rbc package's own exceptions, shared by its downloaders, processors and helpers.
"""


class MissingDataError(Exception):
    """Raised when a data source is missing data that a task is trying to download/process.

    The source doesn't offer it (downloading), or it wasn't downloaded (processing).
    """

    pass


class DataStructureError(Exception):
    """Raised when a source's data isn't what the code expects.

    Its structure, format or vocabulary changed, so the code needs updating.
    """

    pass


class RateLimitError(Exception):
    """Raised when the API rate limit has been reached and downloading must be stopped."""

    pass


class InvalidError(Exception):
    """Raised when provided values/arguments are invalid in some way that forces a stop."""

    pass
