"""Exception hierarchy.

Every error the pipeline raises on bad input derives from
``TranscriberError`` so the CLI can turn it into a one-line message instead
of a traceback.
"""


class TranscriberError(Exception):
    """Base class for all expected, user-facing errors."""


class AudioLoadError(TranscriberError):
    """The input audio file is missing, unreadable or empty."""


class DictionaryError(TranscriberError):
    """A dictionary file or its JSON sidecar is missing or corrupted."""


class DimensionMismatchError(TranscriberError):
    """The dictionary W does not match the spectrogram V."""


class ParameterError(TranscriberError, ValueError):
    """A hyperparameter is outside its valid range."""
