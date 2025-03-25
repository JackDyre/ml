#include "matrix.h"
#include <cassert>
#include <cstdlib>

int main() {
  srand(time(0));

  Matrix m(5, 5);
  m.rand_d(-1, 1);

  m.print_h();

  m.relu_d(m);

  m.print_h();
}
