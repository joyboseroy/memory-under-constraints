from .base import Entry, Memory


class NoMemory(Memory):
    """Never holds anything. With the retrieve policy it only adds read costs."""

    def read(self, key: int) -> Entry | None:
        return None

    def write(self, key: int, value: int, now: int) -> None:
        pass

    def forget(self, key: int) -> None:
        pass
