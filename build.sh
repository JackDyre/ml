#! /bin/sh

export W_FLAGS=(--Werror all-warnings)

set -xe

nvcc -o ./result main.cu "${W_FLAGS[@]}" -arch=sm_89 -O3 --cudart static 

./result
