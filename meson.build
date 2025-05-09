project(
    'ml',
    'cpp',
    'cuda',
    version: '0.0.1',
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

old_kernels_lib = static_library(
    'old_kernels',
    'old_kernels.cu',
    sources: ['old_kernels.h'],
    cuda_args: cuda_args,
)

old_lazy_alloc_lib = static_library(
    'old_lazy_alloc',
    'old_lazy_alloc.cu',
    sources: ['old_lazy_alloc.h'],
    cuda_args: cuda_args,
)

old_matrix_lib = static_library(
    'old_matrix',
    'old_matrix.cpp',
    sources: ['old_matrix.h'],
    link_with: [old_kernels_lib, old_lazy_alloc_lib],
    cpp_args: cpp_args,
)

old_nn_lib = static_library(
    'old_nn',
    'old_nn.cpp',
    sources: ['old_nn.h'],
    link_with: [old_matrix_lib, old_kernels_lib],
    cpp_args: cpp_args,
)

main = executable('ml', 'main.cpp', cpp_args: cpp_args, link_with: [old_nn_lib])

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

dual_slice = static_library(
    'dual_slice',
    'dual_slice.cu',
    sources: ['dual_slice.h'],
    cuda_args: cuda_args,
    link_with: [host_slice, device_slice],
)

kernels = static_library(
    'kernels',
    'kernels.cu',
    sources: ['kernels.h', 'kernel_types.h'],
    cuda_args: cuda_args,
)

matrix = static_library(
    'matrix',
    'matrix.cpp',
    sources: ['matrix.h'],
    cpp_args: cpp_args,
    link_with: [dual_slice, kernels],
)

layer = static_library(
    'layer',
    'layer.cpp',
    sources: ['layer.h'],
    cpp_args: cpp_args,
    link_with: [matrix],
)

nn = static_library(
    'nn',
    'nn.cpp',
    sources: ['nn.h'],
    cpp_args: cpp_args,
    link_with: [layer],
)

run = executable(
    'run',
    'run.cpp',
    cpp_args: cpp_args,
    cuda_args: cuda_args,
    link_with: [nn],
)
