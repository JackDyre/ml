#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>

#define relu(z) ((z) < 0 ? 0 : (z))

#define d_relu(z) ((z) < 0 ? 0 : 1)

void launch_matrix_fill_kernel(float *dst, float val, size_t rows, size_t cols);
void launch_matrix_fill_rand_kernel(float *dst, size_t rows, size_t cols,
                                    float low, float high, unsigned seed);
void launch_matrix_add_kernel(float *dst, float *other, size_t rows,
                              size_t cols);
void launch_matrix_mul_kernel(float *dst, float *a, float *b, size_t rows,
                              size_t cols, size_t inner_size);
void launch_matrix_mse_kernel(float *dst, float *output, float *target,
                              size_t rows, size_t cols);
void launch_matrix_relu_kernel(float *dst, size_t rows, size_t cols);
void launch_matrix_gradient_step_kernel(float *param, float *grad, float lr,
                                        size_t rows, size_t cols);
float calculate_dev_mse_cost(float *output_ptr, float *target_ptr, size_t num_elems);

#endif // KERNELS_H
