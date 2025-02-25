#include <stddef.h>

#ifndef TYPES_H
#define TYPES_H

typedef struct Shape {
  size_t rows;
  size_t cols;
} Shape;

typedef enum Device {
  DEVICE,
  HOST,
  NONE,
} Device;

#endif // TYPES_H
