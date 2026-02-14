"""Real hardware EMG reader via USB serial.

Reads binary packets from an ESP32/Arduino streaming ADC samples.
Plug-and-play replacement for MockReader -- just set EMG_READER=serial in .env.

Packet format:
  Header:    0xAA 0x55 (2 bytes)
  Version:   1 byte
  Seq:       2 bytes (uint16 big-endian)
  Timestamp: 4 bytes (uint32 big-endian, microseconds)
  Channels:  N * 2 bytes (uint16 big-endian per channel)
  CRC16:     2 bytes (optional, currently not validated)

Total: 2 + 1 + 2 + 4 + (N*2) + 2 = 11 + N*2 bytes
For 4 channels: 19 bytes per packet.
"""

import asyncio
import time
from typing import Optional

from emg_core.ingest.base_reader import BaseReader
from emg_core.ingest.packet_parser import PacketParser
from emg_core.api.schemas import RawSample
from emg_core import config


class SerialReader(BaseReader):
    """Reads EMG samples from USB serial (ESP32/Arduino).

    Requires pyserial. Configure via .env:
      SERIAL_PORT=/dev/ttyUSB0
      SERIAL_BAUD=115200
    """

    def __init__(
        self,
        port: str = config.SERIAL_PORT,
        baud: int = config.SERIAL_BAUD,
        num_channels: int = config.NUM_CHANNELS,
    ):
        self.port = port
        self.baud = baud
        self.num_channels = num_channels
        self._serial: Optional[object] = None
        self._parser = PacketParser(num_channels=num_channels)
        self._running = False

    async def connect(self) -> None:
        import serial
        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            timeout=0.1,
        )
        self._running = True

    async def disconnect(self) -> None:
        self._running = False
        if self._serial:
            self._serial.close()  # type: ignore
            self._serial = None

    async def read(self) -> RawSample:
        if not self._running or not self._serial:
            raise RuntimeError("SerialReader not connected. Call connect() first.")

        # Read bytes in a non-blocking way
        while self._running:
            # Read available bytes
            available = self._serial.in_waiting  # type: ignore
            if available > 0:
                data = self._serial.read(available)  # type: ignore
                sample = self._parser.feed(data)
                if sample is not None:
                    return sample

            # Yield to event loop briefly
            await asyncio.sleep(0.001)

        raise RuntimeError("SerialReader stopped.")
