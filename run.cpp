#include "matrix.h"
#include "nn.h"
#include <cassert>
#include <cstdlib>
#include <iostream>

int main() {
  // srand(time(0));
  size_t layer_sizes[] = {2, 4, 2};

  auto nn = NN::from_sizes_slice(layer_sizes, 3);
  nn.random(0, 1);

  Matrix input(2, 1);
  input.fill_d(1);
  nn.set_input(input);
  nn.forward();

  std::cout << "NN Params after forwarding:" << std::endl;
  nn.print();

  auto out = nn.get_output();
  out.print_h();

  auto g = NN::from_sizes_slice(layer_sizes, 3);

  Matrix target(2, 1);
  std::cout << "Target Output:" << std::endl;
  target.fill_d(.5);
  target.print_h();

  nn.back_prop(g, target);

  std::cout << "Gradients:" << std::endl;
  auto o = g.get_output();
  o.print_h();

  g.print();
}
