#include "matrix.h"
#include <cassert>

int main() {
  srand(time(0));

  Matrix m(Shape{.rows = 2, .cols = 2});
  m.rand_d(0, 1);
  m.print_h();

  Matrix n(Shape{.rows = 2, .cols = 2});
  n.rand_d(0, 1);
  n.print_h();

  m.add_d(n);
  m.print_h();
}
