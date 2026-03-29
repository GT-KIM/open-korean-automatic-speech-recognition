# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.

class ConfigParser:
    def __init__(self, entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                value = ConfigParser(value)
            self.__dict__.update({key: value})

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