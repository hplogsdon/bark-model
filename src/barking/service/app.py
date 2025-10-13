import asyncio
import logging
import signal
import time
from time import perf_counter

import pyaudio

from barking.messaging.bus import EventBus

logger: logging.Logger = logging.getLogger(__name__)


async def run(settings: dict) -> None:
    _running = True
    _device = pyaudio.PyAudio()
    _bus = EventBus()

    rate = settings.get("rate", 22050)
    chunk_size = settings.get("chunk_size", 1024)
    interval = settings.get("interval", 10)
    while _running:
        logger.info("start")
        start = time.time()
        stream = _device.open(
            format=settings.get("format", pyaudio.paInt16),
            channels=settings.get("channels", 1),
            rate=rate,
            input=True,
            frames_per_buffer=chunk_size,
        )

        try:
            frames = []
            for i in range(int(rate / chunk_size * interval)):
                data = stream.read(chunk_size)
                frames.append(data)
            stream.stop_stream()
            stream.close()

            await _bus.publish(
                channel="audio",
                type="",
                data={"data": b"".join(frames), "start_ts": start, "end_ts": time.time()},
                headers={},
            )
        except BaseException as e:
            _running = False
            logger.exception("exception", exc_info=e)
        logger.info("ended")


async def stop(sig, loop=None) -> None:
    logger.debug(f"Stopping due to exit signal {sig.name}.")
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for task in tasks:
        logger.debug(f"Stopping {task._coro.__name__}.")
        task.cancel()

    logger.debug(f"Cancelling outstanding tasks. {len(tasks)=}")
    await asyncio.gather(*tasks, return_exceptions=True)


def main():
    logging.info("Starting service.")
    start = perf_counter()
    loop = asyncio.get_event_loop()
    for sig in signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT:
        loop.add_signal_handler(sig, lambda s=sig: loop.create_task(stop(s, loop)))

    loop.run_until_complete(
        run(
            {
                "channels": 1,
                "rate": 22050,
                "chunk_size": 1024,
            }
        )
    )
    end = perf_counter() - start
    logger.info(f"Finished in {end:.2f} seconds.")
