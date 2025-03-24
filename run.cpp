#include "matrix.h"
#include <cassert>

int main() {
  srand(time(0));

  Matrix m(Shape{.rows = 20, .cols = 8});

  m.rand_d(-1, 1);

  m.print_h();
}
