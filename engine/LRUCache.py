from collections import OrderedDict


class LRUCache:
    def __init__(self, maxsize=1024):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, state):
        key = state.__hash__()
        if key in self.cache:
            self.cache.move_to_end(key, last=False)
            return self.cache[key]
        return None

    def put(self, state, value):
        key = state.__hash__()
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem()

    def invalidate(self, state):
        key = state.__hash__()
        self.cache.pop(key)

    def clear(self):
        self.cache.clear()

    def dump(self):
        return dict(self.cache)
