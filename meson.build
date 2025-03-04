project(
    'ml',
    'cpp',
    'cuda',
    version: '0.0.1-unstable',
    default_options: ['warning_level=3'],
)

arch = 'sm_89'


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
)

executable(
    'ml',
    'main.cu',
    cuda_args: cuda_args,
    link_with: [kernels_lib, matrix_lib],
)
