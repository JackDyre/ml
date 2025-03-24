#include "matrix.h"
#include <cassert>

int main() {

  Matrix m(Shape{.rows = 5, .cols = 2});

  m.print_h();

  m.fill_d(2);

  m.print_h();
}
