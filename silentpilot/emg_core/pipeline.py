"""Main EMG pipeline: ingest -> normalize -> segment -> features -> classify.

Orchestrates the full data flow from reader to predictions.
Runs as an async pipeline with an internal sample queue.
"""

import asyncio
import numpy as np
from typing import Optional

from emg_core import config
from emg_core.ingest.base_reader import BaseReader
from emg_core.ingest.mock_reader import MockReader
from emg_core.dsp.normalize import RollingNormalizer
from emg_core.dsp.segment import PTTSegmenter
from emg_core.ml.infer import InferenceEngine
from emg_core.api.schemas import RawSample, Prediction


def create_reader() -> BaseReader:
    """Factory: create the appropriate reader based on config."""
    if config.EMG_READER == "serial":
        from emg_core.ingest.serial_reader import SerialReader
        return SerialReader()
    else:
        return MockReader()


class Pipeline:
    """Async EMG processing pipeline.

    Usage:
        pipeline = Pipeline()
        await pipeline.start()
        sample = await pipeline.get_sample()
        await pipeline.stop()
    """

    def __init__(self):
        self.reader: BaseReader = create_reader()
        self.normalizer = RollingNormalizer()
        self.segmenter = PTTSegmenter()
        self.inference_engine: Optional[InferenceEngine] = None

        self._sample_queue: asyncio.Queue[RawSample] = asyncio.Queue(maxsize=1000)
        self._prediction_queue: asyncio.Queue[Prediction] = asyncio.Queue(maxsize=100)
        self._running = False
        self._read_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the pipeline: connect reader, begin ingestion loop."""
        await self.reader.connect()
        self._running = True
        self._read_task = asyncio.create_task(self._ingest_loop())

    async def stop(self) -> None:
        """Stop the pipeline and clean up."""
        self._running = False
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        await self.reader.disconnect()

    async def _ingest_loop(self) -> None:
        """Continuously read from the reader and push to the sample queue."""
        while self._running:
            try:
                sample = await self.reader.read()

                # Update normalizer (for running stats)
                self.normalizer.update(np.array(sample.ch, dtype=np.float64))

                # Push to queue (drop oldest if full)
                if self._sample_queue.full():
                    try:
                        self._sample_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await self._sample_queue.put(sample)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Pipeline] Ingest error: {e}")
                await asyncio.sleep(0.1)

    async def get_sample(self) -> Optional[RawSample]:
        """Get the next sample from the pipeline (non-blocking)."""
        try:
            return self._sample_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get_prediction(self) -> Optional[Prediction]:
        """Get the next prediction (non-blocking)."""
        try:
            return self._prediction_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def set_inference_engine(self, engine: InferenceEngine) -> None:
        """Set or replace the inference engine."""
        self.inference_engine = engine

    async def classify_segment(self, segment_samples: np.ndarray) -> Optional[Prediction]:
        """Classify a segment using the current inference engine.

        Args:
            segment_samples: 2D array (num_samples, num_channels).

        Returns:
            Prediction if accepted, None otherwise.
        """
        if self.inference_engine is None:
            return None

        prediction = self.inference_engine.predict(segment_samples)
        if prediction:
            await self._prediction_queue.put(prediction)
        return prediction
