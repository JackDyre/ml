#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include <cstddef>
#include <optional>

class Layer {
private:
  Matrix weights;
  Matrix biases;
  Matrix preactivatons;
  Matrix activations;

  std::optional<Layer *> prev;
  std::optional<Layer *> next;

public:
  Layer(std::size_t size, std::size_t prev_size);
  Layer(Matrix weights, Matrix biases, Matrix preactivatons,
        Matrix activations);

  void set_prev(Layer *p);
  void set_next(Layer *n);

  const Matrix get_activations();

  void forward_from(Matrix &prev_acts);
  void forward_chain(Matrix &prev_acts);
};

#endif // !LAYER_H
