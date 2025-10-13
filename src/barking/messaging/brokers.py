import asyncio
from abc import ABC, abstractmethod
from functools import partial
from typing import ClassVar

from barking.messaging.envelope import Envelope
from barking.messaging.message import Message
from barking.messaging.serializers import JSONSerializer, Serializer


class Broker(ABC):
    def __init__(self, url: str, serializer: type[Serializer], node_id: str | None = None, *args, **kwargs):
        if serializer is None:
            self.serializer = JSONSerializer()
        elif type(serializer) is type:
            self.serializer = serializer()
        else:
            self.serializer = serializer
        self.url = url
        self._node_id = node_id

    @property
    def node_id(self) -> str | None:
        return self._node_id

    @node_id.setter
    def node_id(self, node_id: str | None) -> None:
        self._node_id = node_id

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def subscribe(self, channel: str) -> None: ...

    @abstractmethod
    async def unsubscribe(self, channel: str) -> None: ...

    @abstractmethod
    async def publish(self, channel: str, envelope: Envelope) -> None: ...

    @abstractmethod
    async def next(self) -> tuple[str, Envelope]: ...


class MemoryBroker(Broker):
    backend = "memory"

    _channels: ClassVar[set[str]] = set()
    _consumer: asyncio.Queue[tuple[str, Envelope]]

    def __init__(self, *args, max_size: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self._consumer = asyncio.Queue(maxsize=max_size)

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def subscribe(self, channel: str) -> None:
        self._channels.add(channel)

    async def unsubscribe(self, channel: str) -> None:
        if channel in self._channels:
            self._channels.remove(channel)

    async def publish(self, channel: str, message: Message) -> None:
        envelope = Envelope(message)
        envelope.add_nack(partial(self._nack, channel=channel, envelope=envelope))
        envelope.add_ack(partial(self._ack, channel=channel, envelope=envelope))

        await self._consumer.put((channel, envelope))

    async def next(self) -> tuple[str, Envelope]:
        while True:
            channel, envelope = await self._consumer.get()
            if channel in self._channels:
                return channel, envelope

    async def _ack(self, channel: str, envelope: Envelope) -> None:
        self._consumer.task_done()

    async def _nack(self, channel: str, envelope: Envelope) -> None:
        self._consumer.task_done()
        await self._consumer.put((channel, envelope))
