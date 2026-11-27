"""Bounded HTTP range reads for large publisher ZIP releases."""

import io
from http.client import IncompleteRead
from time import sleep
from urllib.request import Request, urlopen


class RangedFile(io.RawIOBase):
    def __init__(self, parts, block_size, *, transfer_limit=None):
        self.parts = parts
        self.size = sum(size for _, size in parts)
        self.block_size = block_size
        self.position = 0
        self.location = None
        self.block = b""
        self.transfer_limit = transfer_limit
        self.requested_bytes = 0

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = (0, self.position, self.size)[whence] + offset
        if self.position < 0:
            raise ValueError("Negative ZIP seek")
        return self.position

    def get_block(self, url, limit, offset):
        end = min(offset + self.block_size, limit) - 1
        span = f"{offset}-{end}"
        request = Request(url, headers={"Range": f"bytes={span}"})
        for attempt in range(3):
            length = end - offset + 1
            if self.transfer_limit is not None and self.requested_bytes + length > self.transfer_limit:
                raise ValueError("Selective ZIP transfer exceeded its byte budget")
            self.requested_bytes += length
            try:
                with urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {span}/{limit}":
                        raise ValueError("Publisher failed the exact HTTP range check")
                    block = response.read(self.block_size + 1)
                if len(block) != length:
                    raise OSError("Truncated ZIP byte range")
                return block
            except (OSError, IncompleteRead):
                if attempt == 2:
                    raise
                sleep(attempt + 1)

    def read(self, size=-1):
        size = min(size if size >= 0 else self.size, max(0, self.size - self.position))
        if size > 64 * 1024**2:
            raise ValueError("ZIP request exceeded the 64 MiB read limit")
        data = bytearray()
        while len(data) < size:
            relative = self.position
            for url, limit in self.parts:
                if relative < limit:
                    break
                relative -= limit
            offset = relative // self.block_size * self.block_size
            if self.location != (url, offset):
                self.block = self.get_block(url, limit, offset)
                self.location = url, offset
            start = relative - offset
            piece = self.block[start:start + size - len(data)]
            data.extend(piece)
            self.position += len(piece)
        return bytes(data)
