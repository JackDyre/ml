project(
    'ml',
    'c',
    'cuda',
    version: '0.0.1-unstable',
    default_options: ['warning_level=3'],
)

arch = 'sm_89'

cuda = find_program('nvcc')

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
    'kernels.h',
    cuda_args: cuda_args,
)

lazy_alloc_lib = static_library(
    'lazy_alloc',
    'lazy_alloc.cu',
    'lazy_alloc.h',
    cuda_args: cuda_args,
)

executable(
    'ml',
    'main.cu',
    link_with: [kernels_lib, lazy_alloc_lib],
    cuda_args: cuda_args,
)
