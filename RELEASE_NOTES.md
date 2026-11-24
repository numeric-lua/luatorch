# luatorch 0.32.0

- five PyTorch API-parity tasks are implemented: `aminmax`, `count_nonzero`,
  `isclose`/`allclose(equal_nan)`, `nan_to_num`, and the
  `torch.linalg.matmul`/`mm`/`mv`/`outer` aliases;
- `torch.aminmax` / `Tensor:aminmax(dim?, keepdim?)` returns the minimum and
  maximum values without materializing indices. A full reduction returns two
  0-dim tensors, an integer dimension returns both reduced shapes, `keepdim`
  keeps them, and NaN propagates through both values;
- `torch.count_nonzero` / `Tensor:count_nonzero(dim?)` returns a 0-dim int64
  count, or counts over an integer or table of distinct dimensions with those
  dimensions removed. NaN counts as non-zero, bool and integer tensors are
  accepted, and an empty/duplicate/out-of-range dimension table raises;
- `torch.isclose` / `Tensor:isclose(other, rtol?, atol?, equal_nan?)` compares
  with broadcasting and defaults `rtol = 1e-5`/`atol = 1e-8`. Finite values
  use `|a - b| <= atol + rtol * |b|`, matching infinities compare close,
  `+inf`/`-inf` do not, and matching NaNs are close only with
  `equal_nan = true`. `allclose` accepts the same flag and now delegates to
  `isclose`, preserving its historical defaults and whole-tensor bool result;
- `torch.nan_to_num` / `Tensor:nan_to_num(nan?, posinf?, neginf?)` replaces
  `NaN`, `+Inf`, and `-Inf`. Floating defaults are zero and the dtype's
  largest/smallest finite value (float16 65504, float32 3.4028e38, float64
  1.7977e308); explicit numbers override individual replacements; the input is
  not modified; integer/bool tensors are cloned unchanged; and gradients pass
  through finite entries with zero at replaced entries;
- `torch.linalg.matmul`, `torch.linalg.mm`, `torch.linalg.mv`, and
  `torch.linalg.outer` mirror `torch.matmul`/`Tensor:matmul`, `mm`, `mv`, and
  `outer`, including broadcasting, dtype promotion, autograd, and clear
  non-tensor validation;
- a NaN read fix: reductions can produce a NaN payload that collides with
  LuaJIT's NaN boxing, causing `get`, `totable`, or `iter` to return `nil` or a
  NULL cdata. Those reads now canonicalize every NaN payload to a Lua NaN, so
  printing and table conversion work for `min`/`max`/`aminmax` results;
- 14 tests in `tests/test_api_compat.lua` cover values, dimensions, keepdim,
  dtype behavior, NaN propagation, `nan_to_num` autograd, tolerance and
  `equal_nan` semantics, infinities, aliases, gradients, and validation
  errors; the suite is now **529 tests** and `./scripts/test.sh` is green;
- docs, docs guards, README status/feature table/build count, roadmap,
  CHANGELOG, and the `0.32.0` rockspec are updated. Every change is pure Lua:
  no C++ shim or FFI declaration changed.

# luatorch 0.31.0

- five real-valued linalg/API gaps are covered: `cholesky_inverse`,
  `cholesky_solve`, the `torch.linalg.lu_unpack` alias, batch-aware
  `matrix_rank` with `hermitian`, and batch-aware `cond` with all real
  orders;
- `torch.linalg.cholesky_inverse` / `Tensor:cholesky_inverse(upper?)` /
  `torch.cholesky_inverse` recovers `A^{-1}` from the lower or upper Cholesky
  factor. Square, batched, and float32/float64 factors are supported, and
  the inverse differentiates through the factor;
- `torch.cholesky_solve` / `Tensor:cholesky_solve(factor, upper?)` solves
  `A X = B` from a Cholesky factor. `B` has shape `(*, n, k)`, its batch
  dimensions may broadcast over the factor, both operands must share
  float32/float64 dtype, and the solution is differentiable. This is the
  top-level PyTorch spelling rather than a new `torch.linalg` name;
- `torch.linalg.lu_unpack(lu, pivots, opts?)` now mirrors
  `torch.lu_unpack`, including the same packed-factor validation and
  `unpack_data`/`unpack_pivots` options;
- `torch.linalg.matrix_rank(a, tol?, hermitian?)` accepts a matrix or batch,
  uses an absolute `tol` when supplied, and switches to absolute symmetric
  eigenvalues with `hermitian=true`. The default tolerance is
  `max(m, n) * eps(dtype) * largest value`; 2-D calls keep returning a Lua
  number while batches return an int64 tensor with the batch shape;
- `torch.linalg.cond(a, p?)` now accepts batches and all real-valued orders:
  `2`, `-2`, `1`, `-1`, `inf`, `-inf`, `"fro"`, and `"nuc"`. The spectral
  orders accept rectangular inputs, `-2` is the reciprocal
  (smallest/largest), and the other orders require square matrices. Plain
  2-D calls keep returning a Lua number;
- 515 tests cover known values, lower/upper factors, batched and empty
  inputs, central-difference gradients, rank tolerances and hermitian paths,
  every condition order, and validation errors; `./scripts/test.sh` is green;
- docs, docs guards, README status/examples/feature table, roadmap, CHANGELOG,
  and the `0.31.0` rockspec are updated. C++ changes are limited to two
  Cholesky shim entry points plus their header and FFI declarations;
  `matrix_rank` and `cond` are implemented in Lua over the existing
  `svdvals`, `eigvalsh`, `matrix_norm`, and `inverse` paths; no timing gate
  was added.

# luatorch 0.30.0

- five real-valued `torch.linalg` gaps are covered:
  `lu_factor_ex`, `solve_ex`, `ldl_factor`, `ldl_factor_ex`, and
  `ldl_solve`;
- `lu_factor_ex` returns the compact packed LU, int32 1-based pivots, and an
  int32 `info` tensor. `check_errors` defaults to false, so singular batches
  are reported instead of raising; `{ check_errors = true }` raises, and the
  unpivoted CPU path is rejected explicitly. Square, rectangular, batched,
  and empty float32/float64 inputs are supported and the factor is
  differentiable;
- `solve_ex` returns the solution and an int32 `info` tensor. `left`
  defaults to true and `check_errors` to false; vector and matrix right-hand
  sides, batch broadcasting, and autograd match the ordinary solve;
- `ldl_factor` exposes compact LDL^T factors through ATen's sytrf path. `LD`
  has the input shape and int32 1-based block pivots have shape `(*, n)`;
  real symmetric square, batched, and empty float32/float64 inputs are
  supported, `hermitian` is accepted, and singular matrices raise;
- `ldl_factor_ex` returns `(LD, pivots, info)` and reports singular batches
  through `info` by default. libtorch 2.5 turns the sytrf positive-info code
  into an internal assertion, so the shim checks `info` itself and raises a
  stable message only when `check_errors=true`;
- `ldl_solve` solves `A X = B` from a compact LDL^T factor. `B` has shape
  `(*, n, k)` with broadcast batch dimensions, pivots are validated as int32
  1-based block indices, and dtype/shape errors are reported in Lua. The LDL
  factorization and solve are value-only because ATen does not implement
  their derivatives;
- 504 tests cover known values, reconstruction identities, rectangular,
  batched, and empty factors, singular `info` diagnostics, autograd for the
  error-reporting LU factor/solve pair, and validation errors;
  `./scripts/test.sh` is green;
- docs, docs guards, README status/examples/feature table, roadmap, CHANGELOG,
  and the `0.30.0` rockspec are updated. C++ changes are limited to five shim
  entry points plus their header and FFI declarations; no timing gate was
  added.

# luatorch 0.29.0

- five real-valued `torch.linalg` gaps are covered: `eigh`/`eigvalsh`,
  `cholesky_ex`, `inv_ex`, the `norm` dispatcher, and a batched
  `matrix_norm` that accepts dimension pairs, `keepdim`, and `"nuc"`;
- `eigh` returns ascending eigenvalues and orthonormal eigenvectors for real
  symmetric square or batched matrices, reads `"L"` by default, accepts
  `"U"`, and exposes `eigvalsh` without materializing eigenvectors. Both
  reject non-square, float16/integer/bool, and invalid `UPLO` inputs;
- `cholesky_ex` and `inv_ex` return an int32 `info` tensor alongside the
  factor or inverse. Successful entries are zero, failures are positive, and
  the result is still returned by default; `check_errors=true` raises the
  LAPACK message instead. Both support batches and empty matrices;
- `torch.linalg.norm` dispatches flattened, vector, and matrix norms like
  PyTorch: no `dim` plus no `ord` gives the flattened 2-norm, explicit `ord`
  without `dim` requires 1-D/2-D input, an integer dimension is a vector norm,
  and a two-element dimension table is a matrix norm. `vector_norm` now
  supports zero, negative finite, and `-inf` orders;
- `matrix_norm` supports every real matrix order (`"fro"`, `"nuc"`, `1`,
  `-1`, `2`, `-2`, `inf`, `-inf`), arbitrary dimension pairs, batches, empty
  batches, and `keepdim`. Plain 2-D calls still return a Lua number;
- 493 tests cover known values, reconstruction identities, lower/upper
  triangles, batch and empty inputs, `info` diagnostics, central-difference
  gradients, all vector/matrix norm orders, dimension pairs, keepdim shapes,
  and validation errors; `./scripts/test.sh` is green;
- docs, docs guards, README status/examples/feature table, roadmap, CHANGELOG,
  and the `0.29.0` rockspec are updated. C++ changes are limited to four shim
  entry points and their header/FFI declarations; no timing gate was added.

# luatorch 0.28.0

- five more real-valued `torch.linalg` APIs are available:
  `torch.linalg.svdvals`, `torch.linalg.vecdot`, `torch.linalg.tensorinv`,
  `torch.linalg.tensorsolve`, and `torch.linalg.householder_product`, each with
  its `Tensor` method and docs/API guards;
- `svdvals` returns descending singular values with shape `(*, min(m, n))`
  through `svd(full_matrices=false)` and differentiates through the values;
- `vecdot` reduces an elementwise product along a dimension with NumPy-style
  broadcasting; the reduction dimension is resolved against the broadcast
  rank, so lower-rank operands work, and float16/float32/float64 are accepted;
- `tensorinv` groups the first `ind` dimensions into a square matrix, inverts
  it, and returns trailing dimensions first; `tensorsolve` solves
  `tensordot(A, X, X:dim()) = B`, optionally moving `dims` of `A` to the end
  first. Both support empty/grouped shapes, matching float32/float64 dtypes,
  and autograd;
- `householder_product` reconstructs `Q` from compact Householder reflectors
  through a minimal native `at::linalg_householder_product` entry point; it
  validates `m >= n`, `k <= n`, batch shapes, and dtype parity, and supports
  batches, empty shapes, and gradients for `A` and `tau`;
- seventeen new tests in `tests/test_linalg_api.lua` cover known values,
  reconstruction identities, grouped and batched shapes, empty inputs,
  central-difference gradients, dtype behavior, and validation errors; the
  suite is now **474 tests** and `./scripts/test.sh` is green;
- docs, README status/examples/feature table, roadmap, CHANGELOG, and the
  `0.28.0` rockspec are updated. C++ changes are limited to one shim entry
  point plus its header and FFI declaration; no timing gate was added.

# luatorch 0.27.0

- five real-valued linear-algebra helpers are now available, implemented in
  Lua over the existing tensor/autograd surface with no native shim changes:
  `torch.lu_unpack` / `Tensor:lu_unpack`, `torch.diagonal` /
  `torch.linalg.diagonal` / `Tensor:diagonal`, `torch.vander` /
  `torch.linalg.vander` / `Tensor:vander`, `torch.cross` /
  `torch.linalg.cross` / `Tensor:cross`, and `torch.linalg.multi_dot`;
- `lu_unpack` rebuilds `P`, `L`, and `U` from `lu_factor`'s packed factor,
  supports square/rectangular/batched/empty matrices, validates int32 1-based
  pivots, exposes `unpack_data`/`unpack_pivots`, and differentiates through
  `L`/`U` while keeping `P` detached;
- `diagonal` supports offsets, arbitrary dimension pairs, batches, and empty
  shapes; `torch.diagonal`/`Tensor:diagonal` default to dimensions 1 and 2,
  while `torch.linalg.diagonal` defaults to the last two dimensions;
- `vander` mirrors both PyTorch forms: `torch.vander` is 1-D and decreasing by
  default, while `torch.linalg.vander` accepts any rank and is increasing by
  default; powers are ordinary differentiable tensor operations;
- `cross` computes real 3-vector products along any dimension, requires matching
  rank/dtype and length 3, broadcasts other dimensions, and carries gradients
  to both inputs; integer and floating dtypes are accepted, bool is rejected;
- `multi_dot` uses a dynamic-programming parenthesization for two or more
  matrices, supports 1-D first/last vector endpoints, requires float16/float32/
  float64 operands, and differentiates through the chain;
- sixteen new tests cover reconstruction, offsets, batches, broadcast shapes,
  vector endpoints, central-difference gradients, option/shape/dtype errors,
  and empty inputs; the suite is now **457 tests**;
- docs, README examples/status/table, API alias guards, CHANGELOG, and ROADMAP
  describe or guard all five helpers. No timing gate was added.

# luatorch 0.26.0

- `torch.linalg.matrix_exp` / `Tensor:matrix_exp()` wraps
  `at::linalg_matrix_exp` and computes the matrix exponential with scaling and
  squaring for square, batched, and empty matrices; the result keeps the input
  shape and dtype;
- only real float32/float64 inputs are accepted. Sub-2-D, non-square, float16,
  integer, and bool inputs are rejected by the Lua layer with clear messages;
  ATen dispatches a CPU half kernel, but it is not numerically reliable (small
  inputs returned NaN or large negative values in a local probe), so half is
  deliberately not exposed;
- five tests cover nilpotent, diagonal, scalar, and rotation generators,
  `exp(A) exp(-A) = I`, `det(exp(A)) = exp(tr A)`, non-contiguous inputs,
  batches and empty matrices, central-difference gradients, the exact
  gradient at zero, and shape/dtype errors; 441 tests total;
- docs, README, design notes, roadmap, CHANGELOG, and the docs test describe
  or guard the new API;
- benchmark entries for 128x128 and 8x32x32 forward passes and a 64x64
  forward+backward pass (no timing gate). The official-bench medians on this
  host were 4.3 ms, 0.45 ms, and 7.9 ms; that suite uses `os.clock` and the
  default thread count, so multi-threaded entries report CPU time rather than
  wall-clock. In a focused one-thread best-of-N check against eager PyTorch
  2.15, matrix_exp was 1.86 vs 1.60 ms for 128x128, 0.46 vs 0.43 ms for
  8x32x32, and 2.44 vs 2.17 ms for the 64x64 forward+backward pass; the
  forward and gradient paths are close to eager, and the cross-version
  comparison is indicative only.

## luatorch 0.25.0

- `torch.linalg.pinv` / `Tensor:pinv(opts?)` wraps `at::linalg_pinv` and
  computes the Moore-Penrose pseudoinverse through the CPU SVD path; it
  accepts square, rectangular, rank-deficient, batched, and empty matrices,
  unlike `inverse`, which requires a square full-rank matrix;
- the pseudoinverse keeps the input dtype (float32/float64) and is
  differentiable with respect to the input; float16, integer, bool, complex,
  and sub-2-D inputs are rejected by the Lua layer with clear messages;
- `opts.atol` and `opts.rtol` are non-negative scalars that set the cutoff
  `max(atol, sigma_1 * rtol)`. With no options ATen uses
  `rtol = max(m, n) * eps`; a positive `atol` without `rtol` sets `rtol` to
  zero, matching PyTorch. Tensor-valued tolerances and the `hermitian` flag
  are not exposed;
- four tests cover full-rank square/tall/wide pseudoinverses, all four
  Moore-Penrose identities for a rank-one matrix, zero/empty/batched inputs,
  relative and absolute cutoffs, central-difference gradients, and
  shape/dtype/option errors;
- docs, README, design notes, roadmap, and CHANGELOG describe the
  pseudoinverse semantics, cutoff defaults, real-dtype limits, and the
  repeated-singular-value autograd caveat; the docs test guards the new
  alias;
- benchmark entries for 128x128 and 256x128 forward and a 64x64
  forward+backward pass (no timing gate). The official-bench medians on this
  host were 8.3-9.0 ms for the 128x128 float32 forward, 12.2-15.4 ms for
  256x128, and 4.0-13.0 ms for the 64x64 forward+backward pass; the suite
  uses `os.clock` with the default thread count and varied by more than 3x
  between runs. In a focused one-thread best-of-N check against eager
  PyTorch 2.15, pinv was 3.07 vs 2.23 ms for 128x128, 3.51 vs 3.32 ms for
  256x128, and 0.84 vs 1.02 ms for the 64x64 forward+backward pass; the
  forward gap tracks `svd` (2.15 vs 1.94 ms), so this workload is close to
  eager PyTorch and the cross-version comparison is indicative only;
- 436 tests.

## luatorch 0.24.0

- `torch.linalg.lu_factor` / `Tensor:lu_factor()` wraps
  `at::linalg_lu_factor_ex` with factorization error checking enabled and
  returns the packed `LU` factor plus int32, 1-based pivots; it supports
  square, rectangular, batched, and empty matrices;
- `torch.linalg.lu_solve` / `Tensor:lu_solve(pivots, b, opts?)` wraps
  `at::linalg_lu_solve` and reuses a factor to solve `A X = B`, `X A = B`
  with `left = false`, or `A^T X = B` with `adjoint = true`;
- only float32/float64 factors and right-hand sides are accepted; singular
  matrices raise at factorization time instead of producing a factor that
  divides by zero later; the Lua layer validates pivot dtype/batch shape,
  square factor shape, right-hand-side layout, and matching dtypes, while
  ATen broadcasts only the right-hand side;
- gradients flow through the packed `LU` and the right-hand side; the
  integer pivots stay detached;
- five tests cover packed square/rectangular factors, batched and empty
  inputs, multiple/right-side/adjoint/broadcast solves, gradients through
  the factor and solve, and validation errors;
- docs, README, design notes, roadmap, and CHANGELOG describe the packed
  representation, 1-based pivots, solve layouts, dtype limits, and autograd
  behavior; the docs test guards both new aliases;
- benchmark entries for factor, reusable solve, and factor+solve
  forward+backward (no timing gate). The official-bench medians on this host
  were 0.15-0.19 ms for a 128x128 float32 factor, 0.08 ms for a 128x128
  factor with 32 right-hand sides, and 3.5 ms for a 64x64 factor+solve
  forward+backward pass; the forward+backward entry is dominated by
  thread/GC variance, and the existing `lu` forward+backward entry moved
  from 0.28 ms to 2.64 ms between two runs of the same suite. In a focused
  one-thread check against eager PyTorch 2.15, the factor was 0.25 vs
  0.13 ms, the reusable solve was 0.19 vs 0.12 ms, and the 64x64
  factor+solve forward+backward pass was 1.25 vs 0.73 ms, so this workload
  is currently slower than eager PyTorch at one thread; the cross-version
  comparison is indicative only;
- 432 tests.

## luatorch 0.23.0

- `torch.linalg.lu` / `Tensor:lu()` wraps `at::linalg_lu` and returns `P`,
  `L`, and `U` with `A = P L U` under partial pivoting;
- square, rectangular, and batched matrices are supported; for an input of
  shape `(*, m, n)` the factors have shapes `(*, m, m)`, `(*, m, k)`, and
  `(*, k, n)` with `k = min(m, n)`;
- float32 and float64 only, matching the CPU LAPACK implementation; other
  dtypes are rejected with clear Lua errors and the unpivoted variant is not
  exposed;
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
