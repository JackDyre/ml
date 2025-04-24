#include <cstddef>
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

typedef struct SubviewArgs {
  size_t rows_from;
  size_t rows_to;
  size_t cols_from;
  size_t cols_to;
} SubviewArgs;

typedef struct IndexSpec {
  std::size_t stride;
  std::size_t ptr_offset;
} IndexSpec;

#endif // TYPES_H
