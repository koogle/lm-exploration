"""Keep an upstream reader open across durable preparation chunks."""

import json

from .scholarly import stream_rows


class PersistentStream:
    def __init__(self, reader=stream_rows):
        self.reader = reader
        self.rows = None
        self.key = None
        self.position = None

    def close(self):
        if self.rows is not None and hasattr(self.rows, "close"):
            self.rows.close()
        self.rows = self.key = self.position = None

    def __call__(self, source, shard, start=0):
        key = (json.dumps(source, sort_keys=True), shard)
        if self.key != key or self.position != start:
            self.close()
            self.rows = iter(self.reader(source, shard, start))
            self.key, self.position = key, start
        # Closing this lightweight generator at a chunk boundary deliberately
        # leaves the underlying stream open. Explicit close releases it.
        while True:
            try:
                row_index, row = next(self.rows)
            except StopIteration:
                self.close()
                return
            except Exception:
                self.close()
                raise
            if row_index != self.position:
                self.close()
                raise ValueError("Upstream row cursor is not contiguous")
            self.position = row_index + 1
            yield row_index, row
