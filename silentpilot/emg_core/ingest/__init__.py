"""EMG signal ingestion - readers for mock and real hardware."""
from .base_reader import BaseReader
from .mock_reader import MockReader

__all__ = ["BaseReader", "MockReader"]
