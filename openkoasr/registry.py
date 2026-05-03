class Registry:
    def __init__(self, name):
        self.name = name
        self._items = {}

    def register(self, key, value):
        if not key:
            raise ValueError(f"{self.name} registry key must be non-empty.")
        self._items[key] = value
        return value

    def get(self, key, default=None):
        return self._items.get(key, default)

    def keys(self):
        return sorted(self._items.keys())

    def as_dict(self):
        return dict(self._items)
