# Changelog

All notable changes to luatorch are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

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
