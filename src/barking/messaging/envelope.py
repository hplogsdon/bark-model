import asyncio
from collections.abc import Awaitable, Callable

from barking.messaging.message import Message


class Envelope:
    def __init__(
        self,
        msg: Message,
        ack: Callable[[], Awaitable[None]] | None = None,
        nack: Callable[[], Awaitable[None]] | None = None,
    ):
        self.msg = msg
        self._ack = ack
        self._nack = nack
        self._lock = asyncio.Lock()
        self._recipient_count = 0
        self._acked_count = 0
        self._nacked = False

    def add_ack(self, func: Callable[[], Awaitable[None]]) -> None:
        self._ack = func

    def add_nack(self, func: Callable[[], Awaitable[None]]) -> None:
        self._nack = func

    async def register_recipient(self) -> None:
        async with self._lock:
            self._recipient_count += 1

    async def ack(self) -> None:
        async with self._lock:
            if self._nacked or self._acked_count >= self._recipient_count:
                return

            self._acked_count += 1
            if self._acked_count == self._recipient_count:
                if self._ack:
                    await self._ack()

    async def nack(self) -> None:
        with self._lock:
            if self._nacked or self._acked_count >= self._recipient_count:
                return

            self._nacked = True
            if self._nack:
                await self._nack()
