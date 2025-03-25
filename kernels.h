#ifndef KERNELS_H
#define KERNELS_H

#include "kernel_types.h"

void device_matrix_fill(MatrixFill args);
void device_matrix_rand(MatrixRand args);
void device_matrix_add(MatrixAdd args);
void device_matrix_mul(MatrixMul args);
void device_matrix_relu(MatrixRelu args);

#endif // !KERNELS_H
