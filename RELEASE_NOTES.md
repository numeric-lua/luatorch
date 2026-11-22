# luatorch 0.23.0

- `torch.linalg.lu` / `Tensor:lu()` wraps `at::linalg_lu` and returns `P`,
  `L`, and `U` with `A = P L U` under partial pivoting;
- square, rectangular, and batched matrices are supported; for an input of
  shape `(*, m, n)` the factors have shapes `(*, m, m)`, `(*, m, k)`, and
  `(*, k, n)` with `k = min(m, n)`;
- float32 and float64 only, matching the CPU LAPACK implementation; other
  dtypes and the unpivoted variant are rejected with clear Lua errors;
- `P` is a permutation matrix and stays detached from autograd; `L` and `U`
  are differentiable;
- four tests cover known square/rectangular factors, row interchanges,
  rectangular/batched/empty inputs, central-difference gradients through `L`
  and `U`, and dimension/dtype errors;
- docs, README, design notes, and the roadmap describe the factorization, its
  output shapes, and dtype/autograd behavior; the docs test guards the new
  alias;
- benchmark entries for forward and forward+backward (no timing gate); the
  best official-bench medians were 0.196 ms for a 128x128 float32
  factorization and 0.270 ms for a 64x64 forward+backward pass. Medians varied
  by up to ~4x across runs while another checkout was compiling on the same
  host. A separate wall-clock check measured 0.28-0.33 ms forward and
  0.31-0.37 ms forward+backward against eager PyTorch 2.15 at 0.17-0.20 ms and
  0.26-0.39 ms, so forward is currently slower and backward is roughly at
  parity; the cross-version comparison is indicative only;
- 427 tests.

## luatorch 0.22.0

- `torch.linalg.solve_triangular` / `Tensor:solve_triangular(b, opts)` wraps
  `at::linalg_solve_triangular` for lower and upper triangular systems,
  including unitriangular solves and right-side (`left = false`) solves;
- `opts.upper` is required so a lower/upper mix-up fails loudly instead of
  silently returning a wrong answer; `left` defaults to true and
  `unitriangular` defaults to false;
- accepts 1-D vector right-hand sides for left solves, matrix right-hand sides
  in `(*, n, k)` / `(*, k, n)` layouts for left/right solves, and broadcast
  batch dimensions; float32 and float64 only, matching CPU LAPACK;
- differentiable with respect to both the triangular matrix and the
  right-hand side; gradients with respect to the unused triangle are zero;
- six tests cover forward semantics, batching, right-side solves, analytic
  gradients, agreement with a full solve, and option/shape/dtype errors;
- docs and README describe the required option, layouts, dtype limits, and
  autograd behavior; the docs test guards the new alias;
- benchmark entries for forward and forward+backward; on this CPU the median
  lower-triangular 128x128 float32 solve is 0.08 ms and the 64x64
  forward+backward pass is 0.18 ms (one thread, no timing gate), roughly at
  parity with eager PyTorch on the same workloads;
- 423 tests.

## luatorch 0.21.0

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
