project(
    'ml',
    'cpp',
    'cuda',
    version: '0.0.1-unstable',
    default_options: ['warning_level=3', 'cpp_std=c++20'],
)

arch = 'sm_89'

cpp_args = ['-std=c++20', '-Wall', '-Wextra', '-Werror', '-O3']

cuda_args = [
    '-arch=' + arch,
    '-O3',
    '--cudart',
    'static',
    '--Werror',
    'all-warnings',
]

kernels_lib = static_library(
    'kernels',
    'kernels.cu',
    sources: ['kernels.h'],
    cuda_args: cuda_args,
)

lazy_alloc_lib = static_library(
    'lazy_alloc',
    'lazy_alloc.cu',
    sources: ['lazy_alloc.h'],
    cuda_args: cuda_args,
)

matrix_lib = static_library(
    'matrix',
    'matrix.cpp',
    sources: ['matrix.h'],
    link_with: [kernels_lib, lazy_alloc_lib],
    cpp_args: cpp_args,
)

nn_lib = static_library(
    'nn',
    'nn.cpp',
    sources: ['nn.h'],
    link_with: [matrix_lib, kernels_lib],
    cpp_args: cpp_args,
)

main = executable(
    'ml',
    'main.cpp',
    cuda_args: cuda_args,
    cpp_args: cpp_args,
    link_with: [nn_lib],
)

determine_arch = executable('determine_arch', 'determine_arch.cu')
