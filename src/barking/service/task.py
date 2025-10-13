import logging
import time
from typing import Any

import pyaudio

from barking.messaging.bus import EventBus


class Task:
    def __init__(self, interval: int):
        self._interval = interval
        self._logger = logging.getLogger(self.__class__.__name__)

    async def run(self) -> None:
        raise NotImplementedError


class AudioTask(Task):
    def __init__(self, bus: EventBus, options: dict[str, Any]):
        self._bus = bus
        self._timeout = options.get("timeout", 60)
        self._format = options.get("format", pyaudio.paInt16)
        self._channels = options.get("channels", 1)
        self._rate = options.get("rate", 22050)
        self._chunk_size = options.get("chunk_size", 1024)

        self._device = pyaudio.PyAudio(0)
        self._device.open()
        super().__init__(options.get("interval"))

    async def run(self) -> None:
        start = time.time()
        stream = self._device.open(
            format=self._format,
            channels=self._channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk_size,
        )

        frames = []
        for i in range(int(self._rate / self._chunk_size * self._chunk_size)):
            data = stream.read(self._chunk_size)
            frames.append(data)
        stream.stop_stream()
        stream.close()

        await self._bus.publish(
            channel="audio",
            type="",
            data={"data": b"".join(frames), "start_ts": start, "end_ts": time.time()},
            headers={},
        )
