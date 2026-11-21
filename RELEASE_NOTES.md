# luatorch 0.21.0

- `torch.linalg.matrix_power` / `Tensor:matrix_power(n)` is a pure-Lua integer
  matrix power using binary exponentiation over `matmul`; it supports square
  and batched square matrices, integer dtypes for non-negative exponents, and
  negative exponents through `inverse` for floating-point inputs;
- `torch.trapezoid` / `Tensor:trapezoid(x?, opts?)` is a pure-Lua trapezoidal
  integration along one 1-based dimension, with uniform `dx`, 1-D or same-shape
  sample positions, and a default dimension of the last one;
- both operations are differentiable (matrix powers through `matmul` and
  `inverse`, trapezoid through its composite arithmetic, including gradients
  with respect to floating-point sample positions);
- ten tests cover matrix powers, trapezoid spacing and shape variants, short
  dimensions, gradients, aliases, and error paths;
- docs and README describe dtype restrictions, option shapes, and autograd
  behavior, and the docs test guards the new alias and function;
- benchmark entries for `matrix_power` and `trapezoid` forward and
  forward+backward (no timing gate); on this CPU `matrix_power` 128x128 p=8 is
  0.17 ms and `trapezoid` 4096 is 0.03 ms in the median run;
- 417 tests.

## luatorch 0.20.0

- `torch.linalg.lstsq` / `Tensor:lstsq(b, opts?)` returns `solution`,
  `residuals`, `rank`, and `singular_values` from `at::linalg_lstsq`;
- supports the `gels`, `gelsy`, `gelsd`, and `gelss` drivers, `rcond` rank
  truncation, batched and multiple right-hand sides, and autograd through the
  solution;
- six linalg tests cover the normal equations, driver outputs, batched and
  underdetermined systems, `rcond` truncation, gradients, and error paths;
- docs describe option, shape, residual, dtype, and driver behavior, with a
  README example and an API alias guard;
- benchmark entries for `lstsq` forward and forward+backward (no timing gate);
- 407 tests.

## luatorch 0.19.0

- `torch.linalg.slogdet` / `Tensor:slogdet()` returns `sign` and `logabsdet`,
  supports batched square matrices, and differentiates through `logabsdet`;
- autograd coverage for einsum traces, batched ellipses, implicit outputs,
  scalar operands, and explicit gradients;
- float16 autograd coverage for elementwise graphs, casts, an `nn.Linear`
  layer, and manual parameter updates;
- predicate masks (`isnan`/`isinf`/`isfinite`) are documented as detached and
  tested with `where` and boolean indexing;
- documentation corrections and checks for linalg aliases and integer/bool
  predicate semantics;
- benchmark entries for einsum, float16 matmul/add, `isfinite`, and `slogdet`;
- 401 tests.
