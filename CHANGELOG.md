# Changelog

All notable changes to luatorch are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- `torch.isposinf` / `Tensor:isposinf()` and `torch.isneginf` /
  `Tensor:isneginf()` return detached bool masks that split infinities by
  sign; NaN and finite values match neither, and integer/bool inputs produce
  all-false masks;
- `torch.nansum` / `Tensor:nansum(dim?, keepdim?)` sums while treating NaNs as
  missing, returns zero for an all-NaN slice, keeps infinities as ordinary
  values, and passes gradients only through non-NaN entries;
- `torch.nanmean` / `Tensor:nanmean(dim?, keepdim?)` averages over the
  non-NaN entries per slice, returns NaN when a slice has no valid entries,
  requires float16/float32/float64, and passes gradients only through non-NaN
  entries;
- `torch.addcmul` / `Tensor:addcmul(t1, t2, value?)` computes
  `self + value * (t1 * t2)` and `torch.addcdiv` / `Tensor:addcdiv` computes
  `self + value * (t1 / t2)`, with broadcasting, a default `value` of 1,
  bool rejection, and autograd through all operands;
- `torch.lerp` / `Tensor:lerp(end, weight)` interpolates
  `self + weight * (end - self)` for float16/float32/float64 inputs, with a
  scalar or broadcastable same-dtype weight and gradients to both endpoints
  and a tensor weight;
- `tests/test_elementwise_extra.lua` covers signed infinities, NaN-aware
  reductions and their gradients, `addcmul`/`addcdiv` broadcasting and
  gradients, `lerp` with scalar/tensor weights, and validation errors.

### Changed

- `docs/api.md` and its docs guards describe the new predicates, reductions,
  and elementwise helpers. All changes are pure Lua; no C++ shim or FFI
  declaration changed.

## [0.34.0] — 2026-11-24

### Added

- `torch.diag_embed` / `Tensor:diag_embed(diagonal?)` embeds the last
  dimension of a tensor on a diagonal of two new trailing dimensions: an input
  of shape `(*, n)` returns `(*, size, size)` with `size = n + |offset|`.
  The 0-based offset defaults to 0, leading dimensions are batch dimensions,
  empty/1-D/2-D/integer/bool inputs are supported, and the result is
  differentiable;
- `torch.diagflat(input, diagonal?)` flattens its input and builds a 2-D
  diagonal matrix from the resulting vector, with the same offset convention
  and autograd behavior as `diag`;
- `torch.tril_indices(row, col, diagonal?, opts?)` and
  `torch.triu_indices(row, col, diagonal?, opts?)` return the selected
  triangle's coordinates as a `2 x N` integer tensor in row-major order.
  Coordinates are 1-based to match the Lua surface; `opts.dtype` selects the
  output dtype (default int64) and `opts.device` must be `"cpu"`;
- `torch.tensordot(a, b, dims?)` contracts the last `dims` dimensions of `a`
  with the first `dims` of `b` (integer form), or arbitrary dimensions via a
  `{a_dims, b_dims}` pair. Paired dimensions must match, both operands must
  share a dtype, free dimensions keep their order, and autograd flows through
  the permute/reshape/matmul implementation;
- `torch.linalg.matrix_rank` accepts an options table
  `{ tol = t, rtol = r, hermitian = h }`. `rtol` uses the relative threshold
  `rtol * largest`; `tol` remains an absolute cutoff, supplying both is an
  error, and the historical numeric/hermitian arguments keep working;
- 15 tests in `tests/test_linalg_build.lua` cover embedded diagonals, batches,
  bool/empty inputs, index coordinates and offsets, tensordot shapes and
  explicit/negative dimensions, autograd for construction and contraction, and
  `matrix_rank` tolerance options; the suite is now **557 tests**.

### Changed

- `docs/api.md`, the docs API guards, README status/feature table/build count,
  roadmap, release notes, and the `0.34.0` rockspec describe or guard the new
  APIs. All changes are pure Lua; no C++ shim or FFI declaration changed.

## [0.33.0] — 2026-11-24

### Added

- `torch.tril` / `Tensor:tril(diagonal?)` and `torch.triu` /
  `Tensor:triu(diagonal?)` return the lower or upper triangular part of a
  matrix or batch of matrices, zeroing the opposite half. The 0-based
  `diagonal` defaults to 0, is positive above the main diagonal and negative
  below it; the mask broadcasts over batch dimensions, every dtype is
  accepted, and autograd flows through the kept triangle with zero gradient in
  the zeroed half;
- `torch.trace` / `Tensor:trace()` sums the main diagonal of a 2-D matrix and
  returns a 0-dim tensor; it accepts integer and floating dtypes and is
  differentiable;
- `torch.diag` / `Tensor:diag(diagonal?)` mirrors `torch.diag`: a 2-D input
  extracts the requested diagonal as a 1-D tensor, while a 1-D input builds a
  square matrix with the vector on that diagonal (padded with zeros). The
  offset uses the same 0-based convention as `tril`/`triu`, empty vectors are
  supported, 1-D construction accepts every dtype including bool, 2-D
  extraction follows `diagonal` (integer and floating dtypes), and both
  directions are differentiable;
- `torch.linalg.solve(a, b, opts?)` and `Tensor:solve(b, opts?)` accept
  `{ left = false }` to solve `X A = B`. The right-hand side then has shape
  `(*, k, n)` and the result keeps that shape; the solve is computed through a
  transposed left solve, so gradients flow to both operands, and vector
  right-hand sides remain left-solve only;
- 13 tests in `tests/test_linalg_triangular.lua` cover values, offsets,
  batches, integer/bool dtypes, empty shapes, autograd, validation, `diag`
  construction and extraction, and the right-side solve; the suite is now
  **542 tests**.

### Changed

- `docs/api.md`, the docs API guards, README status/feature table/build count,
  roadmaps, release notes, and the `0.33.0` rockspec describe or guard the new
  APIs. All changes are pure Lua; no C++ shim or FFI declaration changed.

## [0.32.0] — 2026-11-24

### Added

- `torch.aminmax` / `Tensor:aminmax(dim?, keepdim?)` returns the minimum and
  maximum values without materializing the indices; the full reduction returns
  two 0-dim tensors, an integer dimension returns both reduced shapes, and
  `keepdim` keeps them; NaN propagates through both values;
- `torch.count_nonzero` / `Tensor:count_nonzero(dim?)` counts non-zero entries
  in a 0-dim int64 tensor, or over an integer or table of distinct dimensions
  with the reduced dimensions removed; NaN counts as non-zero;
- `torch.isclose` / `Tensor:isclose(other, rtol?, atol?, equal_nan?)` compares
  two numeric tensors elementwise with broadcasting, defaults
  `rtol = 1e-5`/`atol = 1e-8`, exact equality for matching infinities, and
  optional matching-NaN equality; `allclose` gains the same `equal_nan`
  argument and is now defined through `isclose`, keeping its historical
  defaults and whole-tensor bool result;
- `torch.nan_to_num` / `Tensor:nan_to_num(nan?, posinf?, neginf?)` replaces
  `NaN`, `+Inf`, and `-Inf` with finite values; floating tensors default to
  zero and the dtype's largest/smallest finite value, explicit numbers
  override individual replacements, integer/bool tensors are cloned unchanged,
  and autograd passes through finite entries while replaced entries get zero
  gradient;
- `torch.linalg.matmul`, `torch.linalg.mm`, `torch.linalg.mv`, and
  `torch.linalg.outer` mirror the corresponding `torch`/`Tensor` products,
  including broadcasting, dtype promotion, autograd, and non-tensor validation;
- 14 tests in `tests/test_api_compat.lua` cover values, dimensions, keepdim,
  dtype behavior, warnings/NaN inheritance, autograd for `nan_to_num`,
  `isclose`/`allclose` tolerances, `equal_nan`, infinities, aliases, and
  validation errors; the suite is now **529 tests**.

### Fixed

- Reading a tensor value through `get`, `totable`, or `iter` now canonicalizes
  NaN payloads that collide with LuaJIT's NaN boxing. Reductions such as
  `min`/`max` (and the new `aminmax`) can produce a NaN bit pattern that
  previously read back as `nil` or a NULL cdata; they now return a Lua NaN, so
  printing, table conversion, and iteration work for those results.

### Changed

- `docs/api.md`, the docs API guards, README status/feature table/build count,
  roadmap, release notes, and the `0.32.0` rockspec describe or guard the new
  APIs. All changes are pure Lua; no C++ shim or FFI declaration changed.

## [0.31.0] — 2026-11-24

### Added

- `torch.linalg.cholesky_inverse` / `Tensor:cholesky_inverse(upper?)` /
  `torch.cholesky_inverse` recovers the inverse of a real symmetric
  positive-definite matrix from its lower (`A = L L^T`) or upper
  (`A = U^T U`) Cholesky factor; square, batched, and float32/float64 inputs
  are supported, and the inverse is differentiable;
- `torch.cholesky_solve` / `Tensor:cholesky_solve(factor, upper?)` solves
  `A X = B` from a Cholesky factor, with `B` of shape `(*, n, k)`, broadcast
  batch dimensions, lower/upper factors, shared float32/float64 dtype, and
  autograd through the right-hand side;
- `torch.linalg.lu_unpack(lu, pivots, opts?)` is now exposed as an alias of
  `torch.lu_unpack`, including the namespace and docs guards;
- `torch.linalg.matrix_rank` accepts matrices or batches, an absolute `tol`,
  and `hermitian=true` for real symmetric square inputs; 2-D calls keep the
  historical Lua-number result while batches return an int64 tensor with the
  batch shape, and the default tolerance is
  `max(m, n) * eps(dtype) * largest singular value`;
- `torch.linalg.cond` accepts batches and every real-valued order: `2`, `-2`,
  `1`, `-1`, `inf`, `-inf`, `"fro"`, and `"nuc"` (case-insensitive). The
  spectral orders accept rectangular inputs and `-2` is the reciprocal
  (smallest/largest), while the remaining orders require square matrices;
  2-D calls keep the historical Lua-number result.

### Changed

- `docs/api.md`, the docs API guards, README status/examples/feature table,
  roadmap, release notes, and the `0.31.0` rockspec describe or guard the new
  APIs. C++ changes are limited to two Cholesky shim entry points plus their
  header and FFI declarations; `matrix_rank` and `cond` are pure Lua.
- 515 tests cover the new values, lower/upper factors, batched and empty
  inputs, central-difference gradients, rank tolerances and hermitian paths,
  all condition orders, and validation errors.

## [0.30.0] — 2026-11-23

### Added

- `torch.linalg.lu_factor_ex` / `Tensor:lu_factor_ex(opts?)` returns the
  packed LU, int32 1-based pivots, and an int32 `info` tensor; `pivot`
  defaults to true (the CPU backend rejects unpivoted factorization) and
  `check_errors` defaults to false, so singular batches are reported instead
  of raising. Square, rectangular, batched, and empty float32/float64 inputs
  are supported, and the factor is differentiable;
- `torch.linalg.solve_ex` / `Tensor:solve_ex(b, opts?)` returns the solution
  plus an int32 `info` tensor; `left` and `check_errors` default to
  true/false, vector and matrix right-hand sides are supported with
  broadcast batches, and the solution is differentiable;
- `torch.linalg.ldl_factor` / `Tensor:ldl_factor(opts?)` exposes compact
  LDL^T factors through `at::linalg_ldl_factor_ex`: `LD` keeps the same shape
  as the input and int32 1-based block `pivots` has shape `(*, n)`; real
  symmetric float32/float64 square, batched, and empty inputs are supported,
  singular matrices raise a stable message, and `hermitian` is validated;
- `torch.linalg.ldl_factor_ex` / `Tensor:ldl_factor_ex(opts?)` returns
  `(LD, pivots, info)`; `info` is zero on success and positive for singular
  batches, `check_errors=false` is the default, and `check_errors=true`
  raises. libtorch 2.5 maps the sytrf positive-info code to an internal
  assertion, so the shim checks `info` itself and reports a stable error;
- `torch.linalg.ldl_solve` / `Tensor:ldl_solve(pivots, b, opts?)` solves
  `A X = B` from a compact LDL^T factor and its 1-based block pivots; `B`
  has shape `(*, n, k)` with broadcast batch dimensions, dtype/pivot/shape
  validation matches `lu_solve`, and `hermitian` is accepted for real inputs;
- 504 tests cover the new values, reconstructions, `info` diagnostics,
  rectangular/batched/empty factors, singular handling, gradients for the
  error-reporting LU factor and solve, and validation errors.

### Changed

- `docs/api.md`, the docs API guards, README status/examples/feature table,
  roadmap, release notes, and the `0.30.0` rockspec describe or guard the new
  APIs; C++ changes are limited to five shim entry points plus their header
  and FFI declarations.

## [0.29.0] — 2026-11-23

### Added

- `torch.linalg.eigh` / `Tensor:eigh(uplo?)` and
  `torch.linalg.eigvalsh` / `Tensor:eigvalsh(uplo?)` expose real symmetric
  eigendecomposition through `at::linalg_eigh` and `at::linalg_eigvalsh`;
  eigenvalues are ascending, eigenvectors are orthonormal columns, lower or
  upper triangle selection is validated, and square, batched, and empty
  float32/float64 inputs are supported with autograd;
- `torch.linalg.cholesky_ex` / `Tensor:cholesky_ex(opts?)` returns the
  Cholesky factor plus an int32 `info` tensor, keeps the factor on failed
  factorizations by default, raises with `check_errors=true`, and supports
  `upper`, batches, and empty matrices;
- `torch.linalg.inv_ex` / `Tensor:inv_ex(opts?)` returns the inverse plus an
  int32 `info` tensor, reports singular batch elements without raising by
  default, raises with `check_errors=true`, and differentiates through the
  inverse;
- `torch.linalg.norm` dispatches flattened, vector, and matrix norms from
  `ord` and `dim`, including integer dimensions and one- or two-element
  dimension tables; `vector_norm` gains zero, negative finite, and `-inf`
  orders plus flattened keepdim shaping;
- `torch.linalg.matrix_norm` now accepts batches, arbitrary dimension pairs,
  `keepdim`, and the nuclear norm, while preserving the historical Lua-number
  result for plain 2-D inputs;
- 493 tests cover the new values, reconstructions, UPLO/triangle handling,
  batch and empty inputs, `info` diagnostics, central-difference gradients,
  all norm orders, dimension pairs, keepdim shapes, and validation errors.

### Changed

- `docs/api.md`, the docs API guards, README status/examples/feature table,
  roadmap, release notes, and the `0.29.0` rockspec describe or guard the new
  APIs; C++ changes are limited to four shim entry points plus their header
  and FFI declarations.

## [0.28.0] — 2026-11-23

### Added

- `torch.linalg.svdvals` / `Tensor:svdvals()` returns the descending singular
  values of a matrix or batch through the existing real SVD path, with shape
  `(*, min(m, n))`, float32/float64 validation, and autograd through the
  values;
- `torch.linalg.vecdot` / `Tensor:vecdot(other, dim?)` computes a broadcast
  vector dot product along one dimension, resolving `dim` against the
  broadcast rank, accepting float16/float32/float64, and differentiating
  through both inputs;
- `torch.linalg.tensorinv` / `Tensor:tensorinv(ind?)` groups the first `ind`
  dimensions of a tensor into rows and the rest into columns, inverts the
  square reshaping, and returns the trailing dimensions first; supports empty
  and grouped shapes, real dtypes, and autograd;
- `torch.linalg.tensorsolve` / `Tensor:tensorsolve(b, dims?)` solves
  `tensordot(A, X, X:dim()) = B` for `X`, optionally moving listed dimensions
  of `A` to the end first; supports any `B` shape with the right number of
  elements, empty systems, and gradients with respect to both operands;
- `torch.linalg.householder_product` / `Tensor:householder_product(tau)`
  reconstructs the first `n` columns of a product of Householder reflectors
  from compact `(A, tau)` data, backed by a minimal
  `at::linalg_householder_product` native entry point; validates rank, batch
  shapes, `k <= n`, `m >= n`, and float32/float64 dtype parity, and supports
  batches, empty shapes, and autograd;
- a new `tests/test_linalg_api.lua` file with seventeen tests covering
  values, reconstruction, grouped shapes, batches, empty inputs,
  central-difference gradients, dtype behavior, and validation errors; the
  suite is now **474 tests**.

### Changed

- `docs/api.md` describes the five new APIs, their shapes, dtype rules,
  defaults, and autograd behavior; the docs test guards all five aliases;
- README status, examples, and the feature table, the roadmap, changelog,
  release notes, and the `0.28.0` rockspec name the release. C++ changes are
  limited to the single `lt_linalg_householder_product` entry point and its
  header/FFI declaration.

## [0.27.0] — 2026-11-22

### Added

- `torch.lu_unpack` / `Tensor:lu_unpack(pivots, opts?)` rebuilds `P`, `L`,
  and `U` with `A = P L U` from the packed `(LU, pivots)` returned by
  `lu_factor`; supports square, rectangular, batched, and empty factors,
  validates int32 1-based pivots, exposes `unpack_data`/`unpack_pivots`
  options, and differentiates through `L`/`U` while keeping `P` detached;
- `torch.diagonal` / `torch.linalg.diagonal` / `Tensor:diagonal(offset?,
  dim1?, dim2?)` extracts a diagonal from any dimension pair, with
  `torch.diagonal`/`Tensor:diagonal` defaulting to dimensions 1 and 2 and
  `torch.linalg.diagonal` defaulting to the last two dimensions; offsets,
  batches, arbitrary dimension pairs, empty shapes, and autograd are covered;
- `torch.vander` / `torch.linalg.vander` / `Tensor:vander(N?, increasing?)`
  builds Vandermonde matrices; `torch.vander` is 1-D and decreasing by
  default, while `torch.linalg.vander` accepts any rank and is increasing by
  default, mirroring PyTorch;
- `torch.cross` / `torch.linalg.cross` / `Tensor:cross(other, dim?)`
  computes real 3-vector cross products along any dimension with broadcast
  batch dimensions and autograd through both inputs;
- `torch.linalg.multi_dot({a, b, ...})` multiplies two or more matrices with
  a dynamic-programming parenthesization, including 1-D first/last vector
  endpoints, with autograd through the chain.

### Changed

- docs describe the new real-valued linalg helpers, defaults, validation, and
  autograd behavior; the API documentation tests guard all new aliases; the
  roadmap records them; the README status, feature table, and examples cover
  the additions.

## [0.26.0] — 2026-11-22

### Added

- `torch.linalg.matrix_exp` / `Tensor:matrix_exp()` backed by
  `at::linalg_matrix_exp`, computing the matrix exponential with scaling and
  squaring for square, batched, and empty matrices; the result keeps the input
  shape and dtype;
- Lua validation rejects sub-2-D, non-square, float16, integer, and bool
  inputs with clear messages; only real float32/float64 are exposed because
  ATen's CPU half kernel is not numerically reliable (small inputs can return
  NaN or large negative values);
- five tests covering nilpotent, diagonal, scalar, and rotation generators,
  the identities `exp(A) exp(-A) = I` and `det(exp(A)) = exp(tr A)`,
  non-contiguous inputs, batches and empty matrices, central-difference
  gradients, the exact gradient at zero, and shape/dtype errors;
- benchmark entries for a 128x128 forward pass, an 8x32x32 batch, and a 64x64
  forward+backward pass.

### Changed

- docs describe the matrix exponential, square/batch/empty shapes, real-dtype
  limits, and autograd; the API documentation test guards the new alias; the
  roadmap records the exponential.

## [0.25.0] — 2026-11-22

### Added

- `torch.linalg.pinv` / `Tensor:pinv(opts?)` backed by `at::linalg_pinv`,
  computing the Moore-Penrose pseudoinverse through the CPU SVD path for
  square, rectangular, rank-deficient, batched, and empty matrices; unlike
  `inverse`, it does not require full rank;
- `opts.atol` / `opts.rtol` set the singular-value cutoff
  `max(atol, sigma_1 * rtol)`, with ATen's `max(m, n) * eps` default and
  PyTorch's rule that a positive `atol` alone sets `rtol` to zero;
- Lua validation rejects non-real dtypes (float16, integer, bool), sub-2-D
  inputs, non-table options, unknown keys, and negative or NaN tolerances
  with clear messages; the result keeps the input dtype and is differentiable
  with respect to the input;
- four tests covering full-rank square/tall/wide inverses, all four
  Moore-Penrose identities on a rank-one matrix, zero and empty matrices,
  batched rectangular inputs, cutoff behavior, central-difference gradients,
  and validation errors;
- benchmark entries for square/tall `pinv` and a forward+backward pass.

### Changed

- docs describe pseudoinverse semantics, cutoff defaults, real-dtype limits,
  batch/empty shapes, and the repeated-singular-value autograd caveat; the
  API documentation test guards the new `torch.linalg.pinv` alias; the
  roadmap records the pseudoinverse.

## [0.24.0] — 2026-11-22

### Added

- `torch.linalg.lu_factor` / `Tensor:lu_factor()` backed by
  `at::linalg_lu_factor_ex` with factorization error checking enabled,
  returning the packed `LU` factor and int32, 1-based `pivots`; supports
  square, rectangular, batched, and empty matrices and rejects non-real
  dtypes with clear Lua errors;
- `torch.linalg.lu_solve` / `Tensor:lu_solve(pivots, b, opts?)` backed by
  `at::linalg_lu_solve`, reusing a compact factor to solve `A X = B` with
  `left = false` for `X A = B` and `adjoint = true` for `A^T X = B`;
  validates pivot dtype/batch shape, square factors, right-hand-side layout,
  and float32/float64 dtype, supports batched factors with broadcast
  right-hand sides, and differentiates through `LU` and `B` while keeping
  the integer pivots detached;
- five tests covering packed square/rectangular factors, batched and empty
  inputs, solves from the factor (multiple right-hand sides, right-side and
  adjoint solves, broadcast batches), gradients through the factor and
  solve, and shape/dtype/option errors;
- benchmark entries for `lu_factor`, `lu_solve`, and a forward+backward
  factor/solve pass.

### Changed

- docs describe the packed representation, 1-based pivots, singular-matrix
  error, solve layouts, dtype restrictions, and autograd behavior; the API
  documentation test guards the new `torch.linalg.lu_factor` and
  `torch.linalg.lu_solve` aliases; the roadmap records the reusable solves.

## [0.23.0] — 2026-11-22

### Added

- `torch.linalg.lu` / `Tensor:lu()` backed by `at::linalg_lu`, returning
  `P`, `L`, and `U` with `A = P L U` under partial pivoting; supports square,
  rectangular, and batched matrices, requires float32/float64 (matching CPU
  LAPACK), keeps the permutation `P` detached, and differentiates through
  `L` and `U`;
- four tests covering known square/rectangular factors, row interchanges,
  batches and empty matrices, central-difference gradients through `L` and
  `U`, and dimension/dtype errors;
- benchmark entries for `lu` forward and forward+backward.

### Changed

- docs and README describe the factorization, output shapes, dtype
  restrictions, and autograd behavior; the API documentation test guards the
  new `torch.linalg.lu` alias; the roadmap records `lu` as implemented.

## [0.22.0] — 2026-11-22

### Added

- `torch.linalg.solve_triangular` / `Tensor:solve_triangular(b, opts)` backed
  by `at::linalg_solve_triangular`; `opts.upper` is required, `left` defaults
  to true, `unitriangular` defaults to false, vector and matrix right-hand
  sides plus broadcast batch dimensions are supported, and float32/float64
  inputs are required because CPU LAPACK has no half-precision triangular
  solve;
- six tests covering lower/upper/unitriangular solves, vector and batched
  right-hand sides, right-side (`left = false`) solves, analytic gradients,
  agreement with full `solve` on the stored triangle, and option/shape/dtype
  errors;
- benchmark entries for `solve_triangular` forward and forward+backward.

### Changed

- docs and README describe the required `upper` option, right-hand-side
  layouts, dtype restrictions, and autograd behavior; the API documentation
  test guards the new `torch.linalg.solve_triangular` alias.

## [0.21.0] — 2026-11-21

### Added

- `torch.linalg.matrix_power` / `Tensor:matrix_power(n)`, a pure-Lua integer
  matrix power with binary exponentiation over `matmul`; supports square and
  batched square matrices, integer dtypes for non-negative exponents, negative
  exponents through `inverse` for floating-point inputs, and zero/one
  exponents that return a fresh identity or clone;
- `torch.trapezoid` / `Tensor:trapezoid(x?, opts?)`, a pure-Lua trapezoidal
  integration along one dimension with uniform `dx` or non-uniform 1-D/same-
  shape sample positions, a 1-based `dim` option defaulting to the last
  dimension, and autograd through both values and floating-point samples;
- ten tests covering matrix powers (positive, negative, zero, batched, integer,
  aliases, gradients, validation) and trapezoid (uniform, non-uniform, short
  dimensions, gradients, validation);
- benchmark entries for `matrix_power` forward and forward+backward and for
  `trapezoid` forward and forward+backward.

### Changed

- docs describe both functions, including dtype restrictions, option shape,
  and autograd behavior; the API documentation test guards
  `torch.linalg.matrix_power` and `torch.trapezoid`;
- README shows `matrix_power` and `trapezoid` examples.

## [0.20.0] — 2026-11-21

### Added

- `torch.linalg.lstsq` / `Tensor:lstsq(b, opts?)`, returning solution,
  residuals, rank, and singular values from `at::linalg_lstsq`; supports the
  `gels`, `gelsy`, `gelsd`, and `gelss` drivers, `rcond` rank truncation,
  batched and multiple right-hand sides, and autograd through the solution;
- six `lstsq` tests covering the normal equations, driver outputs, batched and
  underdetermined systems, `rcond` truncation, gradients, and error paths;
- benchmark entries for `lstsq` forward and forward+backward.

### Changed

- docs describe `lstsq` option, shape, dtype, and driver behavior; the API
  documentation test guards the new `torch.linalg.lstsq` alias.

## [0.19.0] — 2026-11-21

### Added

- `torch.linalg.slogdet` / `Tensor:slogdet()`, returning `sign` and
  `logabsdet` from `at::linalg_slogdet`; supports batches, is differentiable
  through `logabsdet`, and reports non-square inputs through ATen;
- autograd coverage for einsum traces, diagonals, batched ellipses, implicit
  outputs, scalar operands, explicit gradients, gradient accumulation, and
  `no_grad`;
- float16 autograd coverage: half-dtype gradients, cast round-trips, an
  `nn.Linear` forward/backward pass, and manual parameter updates;
- predicate autograd coverage: masks stay detached and compose with `where`
  and boolean indexing without leaking gradients through masked values;
- benchmark entries for einsum forward/backward, batched einsum, float16
  matmul and add, `isfinite`, and `slogdet` (no timing gates).

### Changed

- docs describe float16 autograd support plus the CPU LAPACK and optimizer
  dtype limitations;
- documentation tests check that every documented `torch.linalg` alias exists
  and that integer/bool predicate semantics stay all-true.

### Fixed

- corrected the `isfinite` documentation, which claimed integer and bool
  tensors produce all-false masks; they produce all-true masks, matching
  PyTorch and the implementation.

## [0.18.0] — 2026-11-21

### Added

- `torch.isfinite(t)` and `Tensor:isfinite()`, returning bool masks that are
  true for every value that is neither NaN nor +/-Inf;
- `float16` support in `torch.from_blob`, using raw IEEE-754 binary16 buffers
  (`uint16_t` elements, 2 bytes each) without copying;
- einsum edge coverage for ellipsis-only equations, ellipsis matching zero
  dimensions, output ellipsis without input ellipsis, empty batches, 3-D
  diagonals, repeated labels with ellipsis, and empty/scalar operands in every
  position;
- `tests/test_docs.lua` checks version consistency across the API reference,
  rockspec, changelog, release notes, and README, and checks that rockspec
  modules resolve to source files.

### Changed

- einsum rejects mixed operand dtypes and bool operands in the Lua layer with
  a clear message; ATen previously raised `check_type` or `"bmm" not
  implemented` errors. All operands must share one dtype, matching PyTorch;
- `from_blob` distinguishes pointer cdata from arrays by FFI type instead of
  equating an 8-byte object with a pointer, so arrays such as `float[2]` no
  longer require `opts.nbytes`.

## [0.17.0] — 2026-11-21

### Added

- `torch.einsum(equation, ...)` backed by `at::einsum`, with a Lua-side
  validation pass for operand counts, subscripts, ellipses, and output labels;
- `torch.isnan(t)` / `torch.isinf(t)` and `Tensor:isnan()` / `Tensor:isinf()`,
  returning bool tensors (float16/32/64 are checked; integer and bool inputs
  are always false);
- direct `float16` construction from Lua tables and scalars via
  `torch.tensor(..., {dtype = torch.float16})` and `torch.HalfTensor{...}`;
- tests for einsum equations, errors, and autograd; float16 creation,
  rounding, and printing; and the NaN/Inf predicates.

### Changed

- `float16` tensors are now staged as float32 and converted by ATen, which
  applies IEEE round-to-nearest-even and overflow to `inf`; the previous
  "cannot be created from Lua tables" error is gone.

## [0.16.0] — 2026-11-21

### Added

- `torch.linalg.cond` for 2, 1, inf, and Frobenius condition numbers;
- `torch.linalg.matrix_rank` with an optional singular-value tolerance;
- tests for identity, singular matrices, and rank thresholds.

## [0.15.0] — 2026-11-21

### Added

- `matrix_norm` orders `2` (largest singular value) and `-2` (smallest)
  through the existing SVD path;
- tests for spectral norms and unsupported orders.

## [0.14.0] — 2026-11-21

### Added

- `Tensor:take_along_dim(indices, dim)` with 1-based dims/indices;
- `torch.linalg.vector_norm(a[, ord[, dim[, keepdim]]])`;
- tests for gather-aligned selection and vector norm orders.

## [0.13.0] — 2026-11-21

### Added

- `torch.ger` outer-product alias with 1-D validation;
- `Tensor:quantile(q[, dim[, keepdim]])` with linear interpolation;
- `torch.linalg.matrix_norm` for Frobenius, 1, -1, +inf, and -inf norms;
- tests for quantile, outer products, and matrix norm orders.

## [0.12.0] — 2026-11-21

### Added

- `Tensor:median([dim])` for flattened and dimension medians;
- `torch.bincount(input[, weights[, minlength]])`;
- `torch.histc(input[, bins[, min[, max]]])`;
- tests for weighted counts, histogram edges, median shapes, and empty tensors.

### Fixed

- `Tensor:totable()` on an empty tensor now returns an empty table instead of
  raising on a null data pointer.

## [0.11.0] — 2026-11-21

### Added

- `DataLoader` cooperative Lua coroutine workers (`num_workers`), bounded
  `prefetch`, deterministic loader-local `seed`, explicit
  `iterator:close()`/`loader:close()`, and ordered worker-error propagation;
- runnable `examples/dataloader.lua` covering batching, seeded shuffling,
  prefetch, early close, `drop_last`, and empty datasets;
- `docs/data.md` reference for the dataset protocol, loader options, iterator
  lifecycle, worker model, and limits.

### Changed

- `DataLoader:iter()` now returns a callable iterator object; existing
  `for ... in loader:iter()` and `loader:iter()()` usage is unchanged.

### Fixed

- `shuffle = true` on an empty dataset no longer crashes while building an
  empty permutation;
- with `num_workers = 0` and `prefetch > 0`, a look-ahead error now waits for
  earlier prepared batches and is raised at its own batch position, matching
  worker behavior.

### Documentation

- Document that workers are cooperative coroutines in the current Lua state,
  not OS threads or processes, and that they do not overlap with native ATen
  work.

### Tests

- 311 dependency-free tests, including `num_workers = 0` preparation, empty
  datasets across option combinations, exhausted and closed iterators, repeated
  close, and worker/collate error positions.

## [0.10.0] — 2026-11-19

### Added

- SIMD-friendly polynomial `exp`/`tanh`/`sigmoid` helpers in the fused C
  backend; float32 users trade a small accuracy margin for vectorizable math;
- `#pragma omp parallel for simd` on the elementwise kernel.

### Tests

- 303 tests.

## [0.9.0] — 2026-11-19

### Added

- multi-layer RNN/LSTM stacks with PyTorch-compatible `_lN` parameter names;
- bidirectional RNN/LSTM traversal and concatenated outputs;
- inter-layer dropout, disabled in eval mode;
- explicit hidden/cell state tensors per layer and direction;
- 303 total tests.

## [0.8.0] — 2026-11-17

### Added

- PyTorch-style optimizer parameter groups, per-group hyperparameters, and
  `Optimizer:add_param_group`;
- learning-rate schedulers that update every parameter group;
- class weights, `ignore_index`, and label smoothing for
  `CrossEntropyLoss`/`NLLLoss`;
- single-layer `nn.RNN` and `nn.LSTM` with sequence-first and batch-first
  layouts, optional hidden/cell states, and PyTorch parameter names;
- safetensors state-dict interchange with `Module:save_safetensors` and
  `Module:load_safetensors`, including prefix stripping;
- batched `Tensor:index_select_batch` backed by one native C call;
- fused softmax/log-softmax over arbitrary dimensions via a contiguous layout
  move;
- `luatorch-0.8.0-1.rockspec`, `make install`, and `scripts/install.sh`.

### Changed

- `optimizer.lr` mirrors the first parameter group after each step;
- `nn.Softmax`/`nn.LogSoftmax` documentation now covers arbitrary dimensions.

### Tests

- 295 dependency-free tests.

## [0.7.0] — 2026-11-16

- per-tensor-method profiler and release polish;
- direct `from_blob` and fixed Clang build.
