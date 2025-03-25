#include "matrix.h"
#include <cassert>
#include <cstddef>
#include <cstdlib>

std::size_t rand_int_between(std::size_t low, std::size_t high) {
  auto val = (float)low + (float)(high - low) * (float)rand() / (float)RAND_MAX;
  return (std::size_t)val;
}

int main() {
  srand(time(0));

  std::size_t low = 1;
  std::size_t high = 7;

  std::size_t l_dim = rand_int_between(low, high);
  std::size_t r_dim = rand_int_between(low, high);
  std::size_t i_dim = rand_int_between(low, high);

  Matrix l(l_dim, i_dim);
  Matrix r(i_dim, r_dim);
  Matrix d(l_dim, r_dim);

  l.rand_d(-1, 1);
  r.rand_d(-1, 1);

  d.mul_d(l, r);

  d.print_h();
  d.relu_d();
  d.print_h();
}
