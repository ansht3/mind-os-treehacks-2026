"""Abstract base class for EMG signal readers.

This is the plug-and-play interface: implement this to add new hardware.
Switch between readers by setting EMG_READER in .env.
"""

from abc import ABC, abstractmethod
from emg_core.api.schemas import RawSample


class BaseReader(ABC):
    """Abstract EMG signal reader.

    All readers must produce RawSample objects at approximately
    the configured sample rate. The pipeline does not care whether
    samples come from real hardware or a mock generator.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Initialize the reader (open serial port, start generator, etc.)."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up resources."""
        ...

    @abstractmethod
    async def read(self) -> RawSample:
        """Read the next multi-channel sample. Blocks until available."""
        ...
