APP_ABI := arm64-v8a
APP_STL := c++_static
APP_PLATFORM := android-28
APP_CPPFLAGS += -std=c++17 -O3 -Wall -Wextra -fvisibility=hidden -DQNN_API="__attribute__((visibility(\"default\")))"
APP_LDFLAGS += -lc -lm -ldl
