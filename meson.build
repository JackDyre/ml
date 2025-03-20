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
    'old_kernels',
    'old_kernels.cu',
    sources: ['old_kernels.h'],
    cuda_args: cuda_args,
)

lazy_alloc_lib = static_library(
    'old_lazy_alloc',
    'old_lazy_alloc.cu',
    sources: ['old_lazy_alloc.h'],
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
    cpp_args: cpp_args,
    link_with: [nn_lib],
)

determine_arch = executable('determine_arch', 'determine_arch.cu')

ptr = static_library('ptr', 'ptr.cpp', sources: ['ptr.h'], cpp_args: cpp_args)

host_ptr = static_library(
    'host_ptr',
    'host_ptr.cpp',
    sources: ['host_ptr.h'],
    cpp_args: cpp_args,
    link_with: [ptr],
)

device_ptr = static_library(
    'device_ptr',
    'device_ptr.cu',
    sources: ['device_ptr.h'],
    cuda_args: cuda_args,
    link_with: [ptr],
)

slice = static_library(
    'slice',
    'slice.cpp',
    sources: ['slice.h'],
    cpp_args: cpp_args,
)

host_slice = static_library(
    'host_slice',
    'host_slice.cpp',
    sources: ['host_slice.h'],
    cpp_args: cpp_args,
    link_with: [slice, host_ptr],
)

device_slice = static_library(
    'device_slice',
    'device_slice.cpp',
    sources: ['device_slice.h'],
    cpp_args: cpp_args,
    link_with: [slice, device_ptr],
)
