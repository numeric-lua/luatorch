# Design notes

## Goals

1. Run tensor code at native ATen speed from plain LuaJIT.
2. Keep the Torch7 feel (1-based dims, `:add_`, `narrow`, typed constructors)
   without inheriting the old C backend.
3. Make autograd, dtypes, broadcasting, and serialization behave like PyTorch.
4. Make the functional layer luafun-based and Moses-free.
5. Stay small enough to audit and extend.

Non-goals for 0.4: full operator parity with ATen, CUDA, Windows, a complete
`nn` zoo, fused autograd.

## Layers

```
        Lua user code
              │
     lua/torch/init.lua        public module: constructors, aliases, no_grad
     lua/torch/tensor.lua      Tensor class: methods, operators, printing, luafun
     lua/torch/nn/*            modules, losses, containers, functional API
     lua/torch/optim/*         SGD, Adam(W), RMSprop, Adagrad, clipping, schedulers
     lua/torch/utils/*         Dataset, TensorDataset, DataLoader
     lua/torch/fuse.lua        expression tracer + LuaJIT codegen
     lua/torch/fusec.lua       C codegen, gcc driver, kernel cache
     lua/torch/lib.lua         ffi.cdef declarations + shared library loader
              │
        LuaJIT FFI (C ABI only)
              │
     build/libluatorch.so      csrc/luatorch.cpp: exception-safe C wrappers
              │
     libtorch 2.5.1 (CPU)      at::Tensor, autograd, kernels, serialization
```

### Why a C shim instead of FFI to C++?

ATen's public API is C++: `at::Tensor` is a value type with refcounted internals,
operators are overloaded templates, and failures are exceptions. Calling those
symbols through FFI would require mangled names, hand-built STL objects, and
undefined behaviour around exceptions. The shim is ~1k lines of ordinary C++ and
presents:

- opaque `lt_tensor*` handles
- plain C types (`int64_t`, `double`, `void*`)
- status codes + a thread-local error string
- no Lua headers at all, so the native library can be built and tested
  independently of Lua

The FFI layer then costs one indirect call plus argument marshalling per op
(measured at roughly 3 µs for a tiny tensor).

## Memory and lifetimes

- A handle is a heap-allocated `struct lt_tensor { at::Tensor t; }`.
- `at::Tensor` itself is a small refcounted view over storage, so copying a
  handle's tensor is cheap and views (`select`, `transpose`, `slice`) share
  storage exactly as in PyTorch.
- Lua wraps every returned pointer with `ffi.gc(ptr, lt_free)`, so the handle is
  released when the Lua object is collected. Storage is freed when the last
  `at::Tensor` referring to it dies.
- In-place C functions mutate the existing tensor and return the *same* pointer.
  The Lua wrapper ignores that pointer and returns `self`, avoiding a second
  `ffi.gc` registration and a double free.
- A few functions (`grad`, `data_ptr`) return borrowed pointers. The Lua layer
  only uses `data_ptr` while holding a contiguous temporary alive, and `grad`
  wraps a new `at::Tensor` copy of the gradient.

## Errors

C++ exceptions never cross the FFI boundary. Every shim function runs inside a
`try` block; on failure it stores `what()` in a `thread_local std::string` and
returns `NULL` (tensor functions) or a non-zero status. Lua checks the result and
raises `error(last_error())`, preserving the original ATen message (including
shape mismatches, dtype errors, and autograd errors).

## Dtypes and scalar promotion

Dtype ids (`0..8`) are shared between the header, the shim, and `lib.lua`.
Lua has only doubles, which creates one interesting mismatch:

- `torch.tensor({1, 2, 3})` cannot know whether the user meant ints or floats.
  It chooses float32, the PyTorch default floating dtype.
- `long_tensor + 1` *should* stay integral. The shim inspects the tensor dtype
  and, when the value is a whole number, passes an integer `at::Scalar`, avoiding
  PyTorch's float promotion. Non-integral scalars promote as usual
  (`int_tensor / 2` becomes float32, matching true division).
- Integer tensors are explicit: `dtype = torch.int64` or `torch.LongTensor`.

## Autograd

Autograd is ATen's own engine: the shim just forwards `requires_grad`,
`backward`, `grad`, and `detach`. `torch.no_grad(fn)` flips
`at::GradMode::set_enabled` around the call and restores the previous value even
on error. Because grad mode is thread-local in ATen, this is safe if the native
library is ever used from multiple Lua states.

## Indexing model

ATen is 0-based; the Lua API is 1-based. Conversions live exclusively in
`tensor.lua`:

- positive index `i` → `i - 1`; negative index passes through and ATen counts
  from the end;
- index `0` is rejected with a clear message;
- dimensions use the same rule, and `nil` means "all dimensions" via the
  `LT_ALL_DIMS` sentinel;
- `slice(dim, start, stop, step)` has inclusive `start`/`stop` (Torch7 style),
  while ATen's end is exclusive;
- `argmax`/`argmin` add 1 to ATen's result.

## Sorting, indexing and linear algebra

- `sort`/`topk`/`unique` return ATen's 0-based indices; the Lua layer adds one
  so results can be used directly with `index_select`, `get`, or `gather`.
  `unique` returns `values[, inverse, counts]`; the inverse map is also
  shifted to 1-based.
- `gather`/`scatter` accept 1-based index tensors and convert them once at the
  boundary. Index `0` becomes `-1` and fails loudly, which catches off-by-one
  mistakes early.
- `nonzero` returns an `(N, ndim)` int64 tensor of 1-based coordinates.
- Multi-output linear algebra (`svd`, `qr`, `lstsq`) and selection (`sort`,
  `topk`, `unique`) use C out-parameters: the shim writes `lt_tensor*` handles
  and Lua wraps them with finalizers. This avoids tuple/FFI ABI tricks.
- Linear algebra delegates to ATen/LAPACK (`linalg_inv`, `linalg_det`,
  `linalg_slogdet`, `linalg_lstsq`, `linalg_solve`, `linalg_solve_triangular`,
  `linalg_cholesky`, `linalg_svd`, `linalg_qr`), so dtypes and errors match
  PyTorch.
- `solve_triangular` adds Lua-side validation for the required `upper`
  flag, right-hand-side layout, square matrix shape, and float32/float64
  dtype before calling ATen. This prevents a wrong triangle flag from
  silently returning a plausible answer and turns CPU LAPACK's half/integer
  errors into clear messages.
- `multinomial` samples 1-based categories for the same reason as `argmax`;
  uniform/normal/exponential/bernoulli updates are in-place and run through
  ATen's RNG.
- Thread control exposes `at::set_num_threads` / `at::get_num_threads`; OpenMP
  availability is determined by the libtorch build.

## Printing

Printing is implemented in Lua from `totable()` data because formatting needs
Lua's control over precision, dtype suffixes and indentation. Tensors over 1000
elements print a compact summary. No printing happens in C++.

## Neural networks and optimizers

`nn` is deliberately thin. Modules are plain Lua tables (`nn.Module.class`) that
hold parameters and implement `forward`; they call the same Tensor methods
available to users. There is no second autograd engine: `loss:backward()` flows
through ATen's graph, and optimizers update parameters inside `torch.no_grad`.

Design points:

- **Parameter registration** sets `requires_grad` only for floating dtypes,
  matching PyTorch. `self.weight` is a convenience alias for
  `self._parameters.weight` and is refreshed after `module:to(dtype)` replaces
  the parameter objects.
- **Containers** (`Sequential`, `ModuleList`) store children in `_modules`
  with ordered names, which makes `parameters()`, `named_parameters()` and
  `state_dict()` recursive and deterministic.
- **`state_dict` clones** parameters and buffers instead of returning live
  views. This costs memory but prevents a snapshot from silently changing while
  training continues.
- **1-based class labels** are a Lua-surface convention. `argmax` returns
  1-based indices, and `torch.cross_entropy`/`torch.nll_loss` subtract one
  before calling ATen. Torch7 users will recognise `ClassNLLCriterion`
  semantics; PyTorch users see the expected shift.
- **BatchNorm2d** is implemented with tensor reductions (mean, second moment,
  variance) rather than a dedicated C op. Running statistics are updated under
  `no_grad` with the unbiased batch variance, while normalization uses the
  biased variance. A `clamp(0)` guards against tiny negative variances from
  float cancellation.
- **Dropout** builds a Boolean mask with `torch.rand` and scales survivors by
  `1/(1-p)`; it is identity in `eval()`.
- **Serialization** (`nn.save`/`model:save`) writes a small text header per
  tensor followed by contiguous bytes; loading reconstructs detached tensors and
  uses `load_state_dict`, so the architecture is validated (and dtypes cast) on
  the way in.
- **DataLoader** produces batches from an in-memory dataset. `TensorDataset`
  batches with `index_select`; custom datasets are collated by stacking
  per-sample tensors. Shuffling uses `randperm` converted to 1-based indices, or
  a private seeded PRNG when `seed` is set. `num_workers` uses Lua coroutines,
  not OS threads: workers yield one batch per resume and are drained by a
  bounded ordered `prefetch` queue. This keeps dataset access single-threaded,
  preserves order across worker counts, and avoids locks/IPC, but cooperative
  coroutines cannot overlap with native ATen work.
- **Optimizers** keep per-parameter state in a table keyed by tensor identity.
  SGD, Adam, RMSprop and Adagrad follow their PyTorch update equations,
  including Adam bias correction. The update path allocates a few temporaries
  per parameter; fused C kernels are a later optimisation.
- **Lazy package loading** (`torch.nn` / `torch.optim` via `__index`) avoids a
  require cycle when the user starts with `require("torch.nn")`.

## Testing

`tests/run.lua` is a dependency-free harness. The suite covers construction,
math, shape operations, reductions, autograd, luafun integration, serialization,
printing, error paths, every built-in `nn` module/loss, the four optimizers, and
an end-to-end XOR training run. Tests assert both values and metadata (shape/dtype),
and every error test checks a meaningful part of the message.

## Fused execution

`torch.fuse` is deliberately a tracer, not a lazy tensor type. It runs the user
function once with symbolic tensor objects, builds an expression tree, and then:

- emits C and compiles it with `gcc -O3 -march=native -fopenmp` into a shared
  object cached by source hash (`.luatorch-cache`), or
- emits a LuaJIT loop over raw `float*`/`double*` pointers when C is unavailable.

The C ABI is intentionally generic:

```c
void luatorch_kernel(const void** ptrs, const double* scalars,
                     int64_t n, int threads);
```

`ptrs` holds the tensor inputs followed by the output. Scalar arguments are
passed separately, which lets the same compiled kernel serve different scalar
values. OpenMP is enabled only for `n > 20000 && threads > 1`, using libtorch's
thread count.

Two implementation details matter for speed:

- **Branchless activations.** Emitting `v < 0 ? 0 : v` vectorises; `fmaxf`
  without `-ffast-math` becomes a libm call and makes a 1M relu ~4x slower.
- **External-memory-aware GC.** LuaJIT only sees tiny handles, not the 4 MB
  behind each tensor. `tensor.lua` tracks `numel * element_size` and forces a
  full collection when the total crosses 64 MB. Finalizers use
  `lt_free_counted`, which deliberately does not clear the thread-local error
  (otherwise a GC during `ffi.string` can erase an error message being read).

Last-dimension `softmax`, `log_softmax`, and cross-entropy are fixed operations
rather than traced expressions, so their fast-math kernels live in `csrc/fast_ops.cpp`,
compiled with `-ffast-math` while the main shim stays precise. Cross-entropy
reuses the log-softmax kernel, gathers the target log-probability, and uses a
softmax-minus-onehot backward. A C++ autograd Function saves
the log-probabilities and uses `dy - exp(y) * sum(dy)` for the backward.

Fused reductions use a third code shape: an inlined expression helper and four
independent accumulators. The helper is generated from the same expression
tree, so reduction kernels never materialise the elementwise result:

```c
static inline float luatorch_expr(..., int64_t i) { ... }
a0 += luatorch_expr(..., i); ... total = (a0+a1)+(a2+a3);
out[0] = total / n;
```

Reduction kernels opt into `-ffast-math` so GCC can vectorise the accumulator
loops; the cache key includes the flags. The resulting sum is reassociated
(relative error ~1e-5 on 1M elements), which is appropriate for metrics and
losses. Other fusion kernels remain precise.

`torch.fuse.reduce(fn, "sum"|"mean"|"logsumexp")` returns a 0-dim tensor.
`logsumexp` uses a stable max pass followed by a vectorised `exp`/sum pass, and
the generated backward recomputes those statistics to form softmax weights. When inputs
require grad, a reduce backward kernel is generated (`derivative * grad_output`,
divided by `n` for `mean`) and driven by a C++ reduction autograd Function; this
makes fused MSE loss trainable.

Fused training adds a second ABI:

```c
void luatorch_backward(const void** ptrs, const double* scalars,
                       int64_t n, int threads);
// ptrs = [inputs..., grad_output, grad_inputs...]
```

`torch.fuse.train` differentiates the expression symbolically and emits this
kernel. A C++ `torch::autograd::Function` (`lt_fused_apply`) runs the forward
kernel, saves contiguous inputs and scalar arguments in `AutogradContext`, and
calls the backward kernel from `backward()`. This makes the fused result a real
autograd node, so `nn.Linear`, losses, and `loss:backward()` work normally.

The backward code is rendered inline rather than through the forward
temporary-per-node emitter. A shared mask plus inline expressions vectorises;
the earlier temporary-heavy form was ~6x slower because GCC saw the conditional
as control flow. Fused training is first-order and C-backend only; the fixed
cost of the custom node and gradient allocations makes its speedup (~1.3x) more
modest than inference fusion (~5x).

Optimizer updates use the same philosophy: Adam/AdamW/SGD/RMSprop/Adagrad are
single C loops over `data_ptr`, with `__restrict` pointers and `-O3 -march=native
-fno-math-errno`, avoiding the ~10 temporary tensors the Lua implementation
would allocate. Gradient clipping uses the same approach: norm accumulation and
in-place scaling happen without `abs`/`pow`/`sum` temporaries. This makes the optimizer step ~5x faster than PyTorch on the
MLP benchmark.

Unsupported expressions and shape mismatches fall back to eager execution, so
the tracer is never on the correctness path. Inference fusion also falls back
when inputs require grad; the trainable path handles first-order gradients.

## Benchmarks

`bench/bench.lua` measures creation, elementwise throughput, matmul GFLOP/s,
per-call dispatch overhead, Lua-side iteration, luafun pipelines, and
serialization. Results are machine-dependent; the script is the source of truth.

## Adding an operator

1. Declare it in `csrc/luatorch.h` (plain C types only).
2. Implement it in `csrc/luatorch.cpp` inside `try`/`catch`, using `clear_error()`
   and the `LT_CATCH_NULL`/`LT_CATCH_INT` macros. Resolve negative dims with
   `resolve_dim` and indices with `resolve_index`.
3. Add the matching declaration to the `ffi.cdef` block in `lua/torch/lib.lua`.
4. Add a Lua method in `lua/torch/tensor.lua`, converting Lua 1-based
   dims/indices and scalar promotion as needed. Add an `_` variant if ATen
   provides an in-place kernel.
5. Add tests (value, shape, dtype, and at least one error path).
6. Document it in `docs/api.md`.

## Build and distribution

The shim is built out-of-tree with CMake and links the official libtorch
distribution. `third_party/libtorch` is downloaded, not vendored, keeping the
repository small. The build sets an RPATH to the bundled `libtorch/lib` so
`ffi.load` can find ATen's shared libraries without `LD_LIBRARY_PATH`.

Installing (`cmake --install`) copies the Lua tree to `share/lua/5.1` and places
`libluatorch.so` in `share/lua/5.1/torch/`, next to `lib.lua`, which is where the
loader looks last.
