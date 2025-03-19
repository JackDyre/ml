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

typedef enum DeviceStrict {
  _STRICT_DEVICE,
  _STRICT_HOST,
} DeviceStrict;

typedef enum DeviceOpt {
  _OPT_DEVICE,
  _OPT_HOST,
  _OPT_NONE,
} DeviceOpt;

#endif // TYPES_H
