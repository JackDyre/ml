#! /bin/sh

export W_FLAGS=(--Werror all-warnings)

set -xe

meson compile -C out

# nvcc -o ./result main.cu "${W_FLAGS[@]}" -arch=sm_89 -O3 --cudart static

# ./result
