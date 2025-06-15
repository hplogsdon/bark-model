import json
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Serializer(Protocol):
    @staticmethod
    def serialize(data: Any) -> bytes: ...

    @staticmethod
    def deserialize(data: bytes) -> Any: ...


class JSONSerializer(Serializer):
    @staticmethod
    def serialize(data: Any) -> bytes:
        return json.dumps(data).encode()

    @staticmethod
    def deserialize(data: bytes) -> Any:
        return json.loads(data.decode())
