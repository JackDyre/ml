#include "matrix.cu"

int main(void) {
  srand(time(0));

  Matrix m1(Shape{.rows = 2, .cols = 5});
  Matrix m2(Shape{.rows = 5, .cols = 3});

  m1.dev_fill_rand(0, 1);
  m2.dev_fill_rand(0, 1);

  m1.host_print();
  m2.host_print();

  Matrix m3(Shape{.rows = 2, .cols = 3});
  m3.dev_mul(m1, m2);
  m3.host_print();

  return 0;
}
