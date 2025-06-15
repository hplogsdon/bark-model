import asyncio
from collections.abc import AsyncIterator

from barking.messaging.message import Message


class Sentinel(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance

    def __repr__(cls):
        return f"<{cls.__name__}>"

    def __bool__(cls):
        return False


class EndOfStream(metaclass=Sentinel):
    pass


class StreamFinished(Exception): ...


class EventStream:
    def __init__(self, *, maxsize: int = 0):
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._closed = asyncio.Event()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    async def __aiter__(self) -> AsyncIterator[Message]:
        try:
            while msg := await self.get():
                yield msg
        except (StreamFinished, asyncio.CancelledError):
            self._closed.set()

    async def get(self) -> Message:
        if (msg := await self._queue.get()) and not isinstance(msg, EndOfStream):
            self._queue.task_done()
            return msg
        raise StreamFinished

    async def put(self, msg: Message | EndOfStream):
        if not isinstance(msg, Message) and not isinstance(msg, EndOfStream):
            raise ValueError("msg must be a Message")

        await self._queue.put(msg)

    async def close(self):
        await self.put(EndOfStream())
