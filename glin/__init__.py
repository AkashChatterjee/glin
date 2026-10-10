"""glin: instant, exactly-explainable statistical classifiers for AI agents."""
from importlib.metadata import PackageNotFoundError, version

# pyproject.toml is the single source of truth for the version; the release
# workflow stamps it from the git tag, so don't hardcode it here.
try:
    __version__ = version("glin-ml")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"
