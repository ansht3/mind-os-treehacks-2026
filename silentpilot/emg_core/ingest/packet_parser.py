"""Binary packet parser for the serial EMG stream.

Parses the binary protocol from the ESP32/Arduino firmware.
Handles sync, framing, and produces RawSample objects.

Packet format:
  0xAA 0x55 | version(1) | seq(2) | timestamp_us(4) | ch[N](2 each) | crc16(2)
"""

import struct
import time
from typing import Optional

from emg_core.api.schemas import RawSample


HEADER = b'\xAA\x55'
HEADER_LEN = 2
VERSION_LEN = 1
SEQ_LEN = 2
TIMESTAMP_LEN = 4
CRC_LEN = 2


class PacketParser:
    """Stateful binary packet parser.

    Feed raw bytes from serial; it returns RawSample when a complete
    valid packet is found. Handles partial reads and resynchronization.
    """

    def __init__(self, num_channels: int = 4):
        self.num_channels = num_channels
        self._channel_bytes = num_channels * 2
        self._packet_len = (
            HEADER_LEN + VERSION_LEN + SEQ_LEN +
            TIMESTAMP_LEN + self._channel_bytes + CRC_LEN
        )
        self._buffer = bytearray()

    def feed(self, data: bytes) -> Optional[RawSample]:
        """Feed raw bytes, return a RawSample if a complete packet is found."""
        self._buffer.extend(data)

        while len(self._buffer) >= self._packet_len:
            # Look for header
            idx = self._buffer.find(HEADER)
            if idx < 0:
                # No header found, discard all but last byte
                self._buffer = self._buffer[-1:]
                return None

            if idx > 0:
                # Discard bytes before header
                self._buffer = self._buffer[idx:]

            if len(self._buffer) < self._packet_len:
                return None  # need more data

            # Parse packet
            packet = bytes(self._buffer[:self._packet_len])
            self._buffer = self._buffer[self._packet_len:]

            try:
                return self._parse_packet(packet)
            except Exception:
                # Bad packet, continue looking
                continue

        return None

    def _parse_packet(self, packet: bytes) -> RawSample:
        """Parse a complete binary packet into a RawSample."""
        offset = HEADER_LEN

        # Version
        _version = packet[offset]
        offset += VERSION_LEN

        # Sequence number (uint16 big-endian)
        seq = struct.unpack('>H', packet[offset:offset + SEQ_LEN])[0]
        offset += SEQ_LEN

        # Timestamp microseconds (uint32 big-endian)
        _timestamp_us = struct.unpack('>I', packet[offset:offset + TIMESTAMP_LEN])[0]
        offset += TIMESTAMP_LEN

        # Channels (uint16 big-endian each)
        channels: list[int] = []
        for _ in range(self.num_channels):
            ch_val = struct.unpack('>H', packet[offset:offset + 2])[0]
            channels.append(ch_val)
            offset += 2

        # CRC16 (not validated for now -- hackathon speed)
        # _crc = struct.unpack('>H', packet[offset:offset + CRC_LEN])[0]

        return RawSample(
            t=time.time(),
            seq=seq,
            ch=channels,
        )

    def reset(self) -> None:
        """Clear the internal buffer."""
        self._buffer.clear()
