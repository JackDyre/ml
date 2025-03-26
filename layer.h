#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include <cstddef>
#include <memory>
#include <optional>

class Layer {
private:
  Matrix weights;
  Matrix biases;
  Matrix preactivatons;
  Matrix activations;

  std::optional<std::shared_ptr<Layer>> prev;
  std::optional<std::shared_ptr<Layer>> next;

public:
  // Constructors
  Layer(std::size_t size, std::size_t prev_size)
      : Layer(Matrix(size, prev_size), Matrix(size, 1), Matrix(size, 1),
             Matrix(size, 1)) {}
  Layer(Matrix weights, Matrix biases, Matrix preactivatons,
        Matrix activations)
      : weights(weights), biases(biases), preactivatons(preactivatons),
        activations(activations) {}

  // Destructor
  ~Layer() = default;
  // Copy constructor
  Layer(const Layer &other) = default;
  // Copy assignment
  Layer &operator=(const Layer &other) = default;
  // Move constructor
  Layer(Layer &&other) noexcept = default;
  // Move assignment
  Layer &operator=(Layer &&other) noexcept = default;

  void set_prev(std::shared_ptr<Layer> p);
  void set_next(std::shared_ptr<Layer> n);

  const Matrix get_activations();

  void forward(Matrix &prev_acts);
  void forward_chain(Matrix &prev_acts);
};

#endif // !LAYER_H
