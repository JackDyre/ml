-----
$$A_i[r, 0] = f(Z_i[r, 0])$$

$$Z_i[r, 0] = B_i[r, 0] + \sum_k W_i[r, k]A_{i-1}[k, 0]$$

-----

$$C = \sum_r(A_{out}[r, o] - T[r, 0])^2$$

-----

$$\frac{\partial C}{\partial A_{out}[r, 0]} = 2(A_{out}[r, 0] - T[r, 0])$$

-----

$$\frac{\partial C}{\partial B_i[r, 0]}$$

$$= \frac{\partial C}{\partial A_i[r, 0]}
\cdot
\frac{\partial A_i[r, 0]}{\partial Z_i[r, 0]}
\cdot
\frac{\partial Z_i[r, 0]}{\partial B_i[r, 0]}$$

$$= \frac{\partial C}{\partial A_i[r, 0]}
\cdot
f^\prime(Z_i [r, 0])
\cdot
1$$

$$= \frac{\partial C}{\partial A_i[r, 0]}
\cdot
f^\prime(Z_i [r, 0])$$

-----

$$\frac{\partial C}{\partial W_i[r, c]}$$

$$= \frac{\partial C}{\partial A_i[r, 0]}
\cdot
\frac{\partial A_i[r, 0]}{\partial Z_i[r, 0]}
\cdot
\frac{\partial Z_i[r, 0]}{\partial W_i[r, c]}$$

$$= \frac{\partial C}{\partial A_i[r, 0]}
\cdot
f^\prime(Z_i [r, 0])
\cdot
A_{i-1}[c, 0]$$

-----

$$\frac{\partial C}{\partial A_i[r, 0]}$$

$$= \sum_k
\left[
\frac{\partial C}{\partial A_{i+1}[k, 0]}
\cdot
\frac{\partial A_{i+1}[k, 0]}{\partial Z_{i+1}[k, 0]}
\cdot
\frac{\partial Z_{i+1}[k, 0]}{\partial A_i[r, 0]}
\right]$$


$$= \sum_k
\left[
\frac{\partial C}{\partial A_{i+1}[k, 0]}
\cdot
f^\prime (Z_{i+1}[k, 0])
\cdot
W_{i+1}[k, r]
\right]$$

-----

