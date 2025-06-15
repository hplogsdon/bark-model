import asyncio
from abc import ABC, abstractmethod
from typing import ClassVar

from barking.messaging.message import Message
from barking.messaging.serializers import JSONSerializer, Serializer


class Broker(ABC):
    def __init__(self, _serializer: type[Serializer] = JSONSerializer, *args, **kwargs):
        self._serializer = _serializer()

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def subscribe(self, channel: str) -> None: ...

    @abstractmethod
    async def unsubscribe(self, channel: str) -> None: ...

    @abstractmethod
    async def publish(self, channel: str, message: Message) -> None: ...

    @abstractmethod
    async def next(self) -> tuple[str, Message]: ...


class MemoryBroker(Broker):
    _consumer: asyncio.Queue[tuple[str, Message]]
    _channels: ClassVar[set[str]] = set()

    async def connect(self) -> None:
        self._consumer = asyncio.Queue()

    async def disconnect(self) -> None:
        pass

    async def subscribe(self, channel: str) -> None:
        self._channels.add(channel)

    async def unsubscribe(self, channel: str) -> None:
        if channel in self._channels:
            self._channels.remove(channel)

    async def publish(self, channel: str, message: Message) -> None:
        await self._consumer.put((channel, message))

    async def next(self) -> tuple[str, Message]:
        while True:
            channel, message = await self._consumer.get()
            if channel in self._channels:
                return channel, message
