#include "matrix.h"
#include "nn.h"
#include <cassert>
#include <cstdlib>

int main() {
  srand(time(0));
  size_t layer_sizes[] = {2, 2};

  auto nn = NN::from_sizes_slice(layer_sizes, 2);
  nn.random(-1, 1);
  auto g = NN::from_sizes_slice(layer_sizes, 2);

  Matrix input(2, 1);
  input.fill_d(1);

  Matrix target(2, 1);
  target.fill_d(.5);

  while (true) {
    // for (int i = 0; i < 10000; i++) {
    nn.set_input(input);
    nn.forward();

    auto out = nn.get_output();
    out.print_h();

    nn.back_prop(g, target);
    nn.step(g, .01);
  }
}
