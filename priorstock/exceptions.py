"""Project-specific exception types."""


class ConfigurationError(ValueError):
    """Raise when a YAML configuration is missing fields or contains invalid values."""


class DatasetAlignmentError(RuntimeError):
    """Raise when price and news data cannot be aligned into a coherent stock timeline."""


class ArtifactConsistencyError(RuntimeError):
    """Raise when on-disk artifacts are missing required fields or contain inconsistent shapes."""


class TextGenerationValidationError(RuntimeError):
    """Raise when sampled technical-indicator text outputs fail the required quality gates."""


class ExternalServiceError(RuntimeError):
    """Raise when a configured external API returns an invalid or failed response."""
