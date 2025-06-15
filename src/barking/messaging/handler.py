import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, Generic, ParamSpec, TypeVar

from barking.messaging.utils import is_async_callable

P = ParamSpec("P")
R = TypeVar("R")

HandlerType = Callable[..., R | Awaitable[R]]


class MessageHandler(Generic[P, R]):
    def __init__(self, func: HandlerType, channel, msg_type: str = "message"):
        if not callable(func):
            raise TypeError("fn must be callable")

        self._func = func
        self._is_async = is_async_callable(self._func)

        self._name = self._func.__name__
        self.channel = channel
        self.msg_type = msg_type

    @property
    def is_async(self) -> bool:
        return self._is_async

    @property
    def name(self) -> str:
        return self._name

    async def __call__(self, *args: P.args, **kwargs: P.kwargs):
        if not self.is_async:
            loop = asyncio.get_event_loop()
            return loop.run_in_executor(None, partial(self._func, *args, **kwargs))
        else:
            return await self._func(*args, **kwargs)

    def match_message_type(self, msg_type: str) -> bool:
        return self.msg_type == msg_type or self.msg_type == "*"


class HandlerGroup:
    _handlers: dict[str, list[MessageHandler]]

    def __init__(self, handler_cls: type[MessageHandler] = MessageHandler, *args, **kwargs):
        self._handler_cls = handler_cls
        self._handlers = {}

    @property
    def handlers(self):
        return self._handlers

    @property
    def channels(self):
        return self._handlers.keys()

    def add_group(self, group: "HandlerGroup"):
        for _, handlers in self._handlers.items():
            for handler in handlers:
                self.register_handler(handler)

    def get_handlers(self, channel: str, msg_type: str) -> list[MessageHandler]:
        return [handler for handler in self.handlers.get(channel, []) if handler.match_message_type(msg_type)]

    def remove_handler(self, handler: MessageHandler):
        self._handlers[handler.channel].remove(handler)
        if not self._handlers[handler.channel]:
            del self._handlers[handler.channel]

    def register_handler(self, handler: MessageHandler):
        if handler.channel in self._handlers.keys():
            self._handlers[handler.channel].append(handler)
        else:
            self._handlers[handler.channel] = [handler]

    def add_handler(self, handler: HandlerType, channel: str, msg_type: str) -> None:
        self.register_handler(self._handler_cls(handler, channel, msg_type))

    def handler(self, channel: str, msg_type: str) -> Callable[..., Any]:
        def decorator(func: HandlerType) -> HandlerType:
            self.add_handler(handler=func, channel=channel, msg_type=msg_type)
            return func

        return decorator
