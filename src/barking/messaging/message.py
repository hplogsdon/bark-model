from collections import namedtuple
from datetime import UTC, datetime
from typing import Any, TypeVar

MessageType = TypeVar("MessageType", bound="Message")
MessageBase = namedtuple("MessageBase", ["type", "data", "headers"], defaults=[..., ..., {}])


class Message(MessageBase):
    @classmethod
    def create(cls: type[MessageType], type: str, data: Any, headers=None):
        if headers is None:
            headers = {}

        if not headers.get("ts", None):
            headers["ts"] = datetime.now(UTC).isoformat()

        return cls(type=type, data=data, headers=headers)
