import asyncio
import logging
import uuid
from contextlib import asynccontextmanager, suppress
from typing import TypeVar

from barking.messaging.brokers import Broker, MemoryBroker
from barking.messaging.handler import HandlerGroup, MessageHandler
from barking.messaging.message import Message
from barking.messaging.stream import EventStream
from barking.messaging.utils import gather_with_concurrency

logger = logging.getLogger(__name__)

TEventBus = TypeVar("TEventBus", bound="EventBus")


class EventBus:
    _node_id: str
    _broker: Broker

    _lock: asyncio.Lock
    _disconnected: asyncio.Event
    _handler_task: asyncio.Task

    _streams: dict[str, set[EventStream]]
    _handler_tasks: dict[str, set[asyncio.Task]]
    _handler_group: HandlerGroup

    def __init__(
        self, node_id: str | None = None, broker: Broker | None = None, *args, concurrency_limit: int = 10, **kwargs
    ):
        self._node_id = node_id or str(uuid.uuid4())
        self._broker = broker or MemoryBroker()
        self._concurrency_limit = concurrency_limit

        self._lock = asyncio.Lock()
        self._disconnected = asyncio.Event()
        self._disconnected.set()

        self._streams = {}
        self._handler_tasks = {}
        self._handler_group = HandlerGroup()

    async def publish(self, channel: str, type: str, data: dict, headers: dict | None = None, **kwargs) -> None:
        """Publish message on channel

        Args:
            channel (str): channel name
            type (str): message type
            data (dict): message data
            headers (dict): message headers
        """
        await self.publish_message(
            channel=channel,
            message=Message.create(
                type=type,
                data=data,
                headers=headers,
            ),
            **kwargs,
        )

    async def publish_message(self, channel: str, message: Message, **kwargs) -> None:
        if not message.headers.get("source"):
            message.headers["source"] = self._node_id

        await self._broker.publish(channel, message, **kwargs)

    @asynccontextmanager
    async def subscribe(self, channel: str, **kwargs) -> None:
        self._ensure_connection()

        stream = EventStream(**kwargs)
        await self._register_stream(channel, stream)
        try:
            yield stream
        finally:
            await stream.close()
            await self._deregister_stream(channel, stream)

    async def connect(self):
        async with self._lock:
            if not self.connected:
                # connect broker
                await self._broker.connect()

                # create readers for each channel in group
                for channel in self._handler_group.channels:
                    await self._create_handler_reader(channel)

                # create reader task
                self._handler_task = asyncio.create_task(self._handler())

                # set connected flag
                self._disconnected.clear()
                await asyncio.sleep(1)

    async def disconnect(self):
        async with self._lock:
            await self.close_streams()

            # cancel tasks
            if not self._handler_task.done():
                self._handler_task.cancel()

            # clean out result
            with suppress(asyncio.CancelledError):
                await self._handler_task

            await self._broker.disconnect()
            self._disconnected.set()

    @property
    def connected(self) -> bool:
        return not self._disconnected.is_set()

    async def create_handler_reader(self, channel):
        async with self._lock:
            await self._create_handler_reader(channel)

    async def register_stream(self, channel: str, stream: EventStream) -> None:
        async with self._lock:
            await self._register_stream(channel, stream)

    async def deregister_stream(self, channel: str, stream: EventStream) -> None:
        async with self._lock:
            await self._deregister_stream(channel, stream)

    async def close_streams(self) -> None:
        for streams in self._streams.values():
            for stream in streams:
                await stream.close()

    async def __aenter__(self: TEventBus) -> TEventBus:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    def _ensure_connection(self):
        if not self.connected:
            raise RuntimeError("Must connect before interaction")

    async def _handler(self):
        """Iteratively add the incoming messages to their respective streams"""
        while True:
            channel, msg = await self._broker.next()
            streams = self._streams.get(channel, set()) | self._streams.get("*", set())
            for stream in streams:
                await stream.put(msg)

    async def _create_handler_reader(self, channel):
        self._handler_tasks[channel] = asyncio.create_task(self._read_stream(channel))

    async def _subscribe_channel(self, channel: str):
        await self._broker.subscribe(channel)

    async def _unsubscribe_channel(self, channel: str):
        await self._broker.unsubscribe(channel)

    async def _register_stream(self, channel: str, stream: EventStream) -> None:
        if not self._streams.get(channel):
            await self._subscribe_channel(channel)
            self._streams[channel] = {
                stream,
            }
        else:
            self._streams[channel].add(stream)

    async def _deregister_stream(self, channel: str, stream: EventStream) -> None:
        if stream_set := self._streams.get(channel, set()):
            # remove the stream set
            stream_set.remove(stream)
            # and cleanup if it was the last one
            if not stream_set:
                del self._streams[channel]
                await self._unsubscribe_channel(channel)

    async def _read_stream(self, channel: str) -> None:
        async with self.subscribe(channel) as stream:
            async for msg in stream:
                await self._fire_handlers(self._handler_group.get_handlers(channel, msg.type), msg)

    async def _fire_handlers(self, handlers: list[MessageHandler], msg: Message):
        return await gather_with_concurrency(
            self._concurrency_limit, *(self._execute_handler(handler, msg) for handler in handlers)
        )

    async def _execute_handler(self, handler: MessageHandler, msg: Message):
        try:
            await handler(msg)
        except BaseException as exc:
            logger.error(f"Encountered exception in Message Handler: {exc}", exc_info=exc)
