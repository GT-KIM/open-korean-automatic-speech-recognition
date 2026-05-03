# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.

class ConfigParser:
    def __init__(self, entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                value = ConfigParser(value)
            self.__dict__.update({key: value})

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigParser):
                value = value.to_dict()
            result[key] = value
        return result

    def get(self, key, default=None):
        return getattr(self, key, default)

if __name__ == "__main__":
    dummy_config = ConfigParser(
        {
            "foo": "bar",
            "data": [0, 1, 2],
            "name": {"man": "John", "woman": "Annie"}
        }
    )
    print(dummy_config.foo)
    print(dummy_config.data)
    print(dummy_config.name.man)
