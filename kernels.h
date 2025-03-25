#ifndef KERNELS_H
#define KERNELS_H

#include "kernel_types.h"

void device_matrix_fill(MatrixFill args);
void device_matrix_rand(MatrixRand args);
void device_matrix_add(MatrixAdd args);

#endif // !KERNELS_H
