from collections import OrderedDict
from chess.polyglot import zobrist_hash


class LRUCache:
    def __init__(self, maxsize=1024):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, state):
        key = zobrist_hash(state)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, state, value):
        key = zobrist_hash(state)
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

    def invalidate(self, state):
        key = zobrist_hash(state)
        self.cache.pop(key)

    def clear(self):
        self.cache.clear()

    def dump(self):
        return dict(self.cache)
