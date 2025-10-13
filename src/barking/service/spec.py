from abc import ABC


class TaskSpec(ABC):
    def __init__(self, device: int = 0):
        self.device = device

    def to_dict(self):
        return {
            "device": self.device,
            "timeout": 10,
            "format": "paInt16",
            "channels": 1,
            "rate": 22050,
            "chunk_size": 1024,
        }
