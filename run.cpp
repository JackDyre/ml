#include "matrix.h"
#include <cassert>

int main() {
  srand(time(0));

  Matrix l(Shape{.rows = 3, .cols = 5});
  Matrix r(Shape{.rows = 5, .cols = 4});
  Matrix d(Shape{.rows = 3, .cols = 4});

  l.rand_d(0, 1);
  r.rand_d(0, 1);

  d.mul_d(l, r);

  l.print_h();
  r.print_h();
  d.print_h();
}
