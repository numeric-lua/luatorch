# luatorch

**A LuaJIT tensor engine with a fusion compiler — PyTorch-class speed with Torch7 ergonomics.**

Torch7 stopped in 2018. PyTorch kept going, but its eager Python API pays for
every operator with dispatch, temporary allocations, and interpreter overhead.
`luatorch` keeps PyTorch's battle-tested ATen kernels and BLAS underneath, then
adds the part Python does not give you cheaply: **a tracing fusion compiler for
LuaJIT**.

```lua
local torch = require("torch")

local fused = torch.fuse(function(a, b, c)
  return (a * b + c):relu()
end)

local out = fused(a, b, c)          -- one vectorized kernel, no temporaries
fused:into(buffer, a, b, c)         -- zero-allocation variant

-- Trainable variant: forward and backward are compiled, then wired into
-- PyTorch's autograd engine through a C++ custom Function.
local train_fused = torch.fuse.train(function(a, b, c)
  return (a * b + c):relu()
end)
local y = train_fused(a, b, c)
y:sum():backward()                  -- compiled backward kernel
```

This is not a thin wrapper around PyTorch. The execution layer:

- traces elementwise tensor expressions and compiles them to **one C kernel**
  (or a LuaJIT loop when no C compiler is available),
- generates **forward and backward** kernels and connects them to PyTorch's
  autograd engine for first-order training,
- eliminates intermediate tensors and memory traffic,
- accounts for C++ tensor memory so LuaJIT's GC does not let results pile up,
- exposes reusable output buffers (`fused:into`) and in-place ops,
- falls back to eager ATen when an expression cannot be fused.

## Measured against PyTorch 2.15 (CPU, 1 thread)

Best-of-30 microbenchmarks from `scripts/compare.sh --compile` on an 8-core
Ryzen 3 machine. Eager and compiled PyTorch are shown separately because
`torch.compile` also fuses.

| Workload | luatorch | torch eager | speedup | torch.compile | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `add` 1M, out-of-place | 0.81 ms | 0.83 ms | **1.03x** | — | — |
| `add_` 1M, in-place | 0.55 ms | 0.55 ms | **1.00x** | — | — |
| `a*b+c` 1M | 1.57 ms | 4.78 ms | **3.05x** | — | — |
| `(a*b+c):relu()` 1M eager | 2.17 ms | 5.53 ms | **2.55x** | 1.05 ms | 0.48x |
| `(a*b+c):relu()` **fused, into** | **1.09 ms** | 5.53 ms | **5.07x** | 1.05 ms | 0.96x |
| `(a*b+c):relu()` **fused, alloc** | **1.03 ms** | 5.53 ms | **5.36x** | 1.05 ms | **1.01x** |
| `sum` 1M | 0.11 ms | 0.14 ms | **1.20x** | — | — |
| `exp` 1M | 3.02 ms | 3.02 ms | **1.00x** | — | — |
| `matmul 512` | 7.66 ms | 7.67 ms | **1.00x** | — | — |
| tiny add (16) | 0.002 ms | 0.003 ms | **1.43x** | — | — |
| tiny chain (16) | 0.006 ms | 0.007 ms | **1.16x** | — | — |
| tiny chain fused (16) | 0.006 ms | 0.007 ms | **1.25x** | — | — |
| MLP forward 128×784 | 1.55 ms | 1.53 ms | **0.99x** | — | — |
| MLP forward+backward | 3.60 ms | 3.62 ms | **1.01x** | — | — |
| **MLP Adam step** | **0.224 ms** | 1.137 ms | **5.08x** | — | — |
| **MLP train step** | **3.97 ms** | 5.12 ms | **1.29x** | — | — |
| chain train step, fused | 11.6 ms | 14.6 ms | **1.26x** | — | — |
| MSE reduce, fused | 0.45 ms | 1.74 ms | **3.87x** | — | — |
| chain reduce, fused | 0.61 ms | 4.04 ms | **6.57x** | — | — |
| MSE loss train, fused | 4.36 ms | 11.55 ms | **2.65x** | — | — |
| logsumexp, fused | 1.30 ms | 4.62 ms | **3.56x** | — | — |
| logsumexp train, fused | 7.3 ms | 13.0 ms | **1.78x** | — | — |
| log_softmax 4096×256, fused | 1.21 ms | 2.11 ms | **1.74x** | — | — |
| log_softmax + backward, fused | 3.50 ms | 4.78 ms | **1.37x** | — | — |
| cross_entropy 4096×256, fused | 4.18 ms | 5.33 ms | **1.28x** | — | — |
| softmax 4096×256, fused | 1.20 ms | 1.63 ms | **1.36x** | — | — |
| softmax + backward, fused | 3.33 ms | 3.82 ms | **1.15x** | — | — |

What the table actually says:

- **Fused elementwise chains** are 2.5–5.4x faster than eager and around parity
  with `torch.compile` on this machine (0.96–1.01x; treat a few percent as
  noise).
- **Optimizer updates are a clear win:** fusing Adam's update into one C kernel
  makes the optimizer step **5.1x faster** than PyTorch's optimizer, which lifts
  the whole MLP training step to **1.29x**.
- **Fused training is a moderate win.** A compiled forward+backward chain runs
  about 1.2–1.5x faster than eager PyTorch in best-of runs, less than the
  inference speedup because the custom autograd node and gradient allocations
  have fixed overhead.
- **Softmax is fused for last-dimension layouts.** The store-then-scale kernel
  reaches **1.36x** forward and **1.15x** through backward; the backward is
  `y * (dy - sum(dy*y))`.
- **Cross-entropy is fused for typical class counts.** It reuses the fast
  log-softmax, computes NLL directly, and backpropagates through a fast
  softmax-minus-onehot kernel: **1.28x** on a 4096×256 classifier. Very large
  class counts (>4096) fall back to ATen, which wins there.
- **Log-softmax is fused for last-dimension softmax layouts.** A fast-math
  kernel computes max/sum/output in one call and a custom autograd Function
  handles the backward, giving **1.7x** forward and **1.37x** forward+backward.
- **Fused reductions are a major win.** Compiling reduction kernels with
  `-ffast-math` lets GCC vectorise the accumulators and libm calls: MSE drops to
  0.45 ms (**3.9x**), a two-op ReLU chain to 0.61 ms (**6.6x**),
  `logsumexp` to 1.30 ms (**3.6x**), and trainable MSE to **2.65x** faster than
  eager. The trade-off is reassociated summation: relative error is ~1e-5 on 1M
  elements, well inside metric/loss tolerance.
- **Single additions and GEMM are parity-to-tie.** GEMM uses the same BLAS-class
  kernels; single out-of-place adds are dominated by allocation/free behavior.
- **Small tensors** are 1.1–1.4x faster thanks to lower dispatch overhead.
- Benchmark variance is real: this CPU throttles under sustained load. The
  harness uses best-of-N and one thread; rerun it for your machine.

## Shared buffers

`torch.from_blob(buffer, shape[, opts])` builds a zero-copy tensor over a LuaJIT
FFI array or a Lua string; the tensor retains the backing buffer. FFI arrays are
mutable and shared in both directions. String-backed tensors should be treated
as read-only. Every dtype is accepted, including `float16`, whose buffers store
raw IEEE-754 binary16 elements (2 bytes each).

## Linear algebra

`torch.linalg` wraps ATen's dense linear algebra. `slogdet` returns the sign and
log-absolute determinant without forming the determinant, so it stays finite for
very large or small magnitudes:

```lua
local a = torch.tensor({{2, 1}, {1, 3}}, {requires_grad = true})
local sign, logabsdet = torch.linalg.slogdet(a)
print(sign:item(), logabsdet:item())   -- 1, 1.6094379425049
logabsdet:backward()                   -- gradient d log|det A| / dA = A^-T
```

`torch.linalg.lstsq` solves dense least-squares problems and returns the
solution, residual sum of squares, numerical rank, and singular values:

```lua
local a = torch.tensor({{1, 0}, {0, 1}, {1, 1}}, {dtype = torch.float64})
local b = torch.tensor({1, 2, 3}, {dtype = torch.float64})
local x, residuals, rank = a:lstsq(b, {driver = "gelsd"})
print(x:totable(), rank:item())   -- {1, 2}, 2
```

See [docs/api.md](docs/api.md) for driver and `rcond` semantics; the solution
is differentiable.

`torch.linalg.solve_triangular` solves a triangular system and requires an
explicit `upper` flag, so a lower/upper mix-up cannot silently return a wrong
answer. Vector and matrix right-hand sides are supported, and gradients flow
to both the matrix and the right-hand side:

```lua
local a = torch.tensor({{2, 0}, {1, 3}}, {dtype = torch.float64})
local b = torch.tensor({2, 4}, {dtype = torch.float64})
print(a:solve_triangular(b, {upper = false}):totable())  -- {1, 1}
```

`torch.linalg.lu` returns the pivoted factorization `A = P L U` for square,
rectangular, and batched matrices. `P` is a detached permutation matrix, while
`L` and `U` carry gradients:

```lua
local a = torch.tensor({{0, 1}, {1, 0}}, {dtype = torch.float64})
local p, l, u = a:lu()
print(p:matmul(l):matmul(u):totable())  -- {{0, 1}, {1, 0}}
```

`torch.linalg.lu_factor` returns a compact factorization and 1-based pivots
that `torch.linalg.lu_solve` can reuse for one or many right-hand sides.
Singular matrices raise at factorization time, and solutions carry gradients
to the factor and the right-hand side:

```lua
local a = torch.tensor({{4, 3}, {2, 1}}, {dtype = torch.float64})
local lu, pivots = a:lu_factor()
local b = torch.tensor({{1}, {2}}, {dtype = torch.float64})
print(lu:lu_solve(pivots, b):totable())  -- {{2.5}, {-3}}
```

`torch.linalg.pinv` computes the Moore-Penrose pseudoinverse, so it also
handles rectangular and rank-deficient matrices where `inverse` raises.
`atol`/`rtol` control which singular values are discarded, and gradients flow
to the input:

```lua
local a = torch.tensor({{1, 0}, {0, 1}, {1, 1}}, {dtype = torch.float64})
print(a:pinv():sizes())                       -- {2, 3}
print(a:pinv({rtol = 1e-6}):sizes())          -- {2, 3}
```

`torch.linalg.matrix_exp` computes the matrix exponential of a square matrix
or a batch of square matrices. It is the true matrix exponential, not an
elementwise `exp`, so it satisfies `exp(A) exp(-A) = I`; gradients flow to the
input:

```lua
local a = torch.tensor({{0, 1}, {0, 0}}, {dtype = torch.float64})
print(a:matrix_exp():totable())               -- {{1, 1}, {0, 1}}
```

`torch.linalg.matrix_power` raises a square matrix to an integer power, and
`torch.trapezoid` integrates a tensor along one dimension with the trapezoidal
rule. Both support autograd and batched `matrix_power`:

```lua
local a = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64,
  requires_grad = true})
local a2 = torch.linalg.matrix_power(a, -2)   -- A^-2
print(a2:totable())                           -- {{5.5, -2.5}, {-3.75, 1.75}}

local y = torch.tensor({1, 2, 3}, {dtype = torch.float64})
print(torch.trapezoid(y, {dx = 0.5}):item())  -- 2.0
```

`torch.linalg.cholesky_inverse` and `torch.cholesky_solve` reuse a Cholesky
factor for a symmetric positive-definite system. The factor is lower by
default; pass `true` for the upper factor. Both support batches and autograd:

```lua
local a = torch.tensor({{4, 2}, {2, 3}}, {dtype = torch.float64})
local b = torch.tensor({{1}, {2}}, {dtype = torch.float64})
local l = a:cholesky()
print(torch.linalg.cholesky_inverse(l):allclose(a:inverse()))  -- true
print(torch.cholesky_solve(b, l):allclose(a:solve(b)))         -- true
```

Smaller matrix helpers round out the same real-valued API. `lu_unpack`
rebuilds `P`, `L`, and `U` from a packed factorization, `diagonal` extracts any
offset from any dimension pair, `vander` builds Vandermonde matrices,
`cross` computes real 3-vector products, and `multi_dot` multiplies a matrix
chain in an optimal order. They compose ordinary differentiable tensor
operations, so autograd works throughout:

```lua
local a = torch.tensor({{4, 3}, {2, 1}}, {dtype = torch.float64})
local lu, pivots = a:lu_factor()
local p, l, u = torch.lu_unpack(lu, pivots)     -- A = P L U

local m = torch.tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
  {dtype = torch.float64})
print(m:diagonal(1):totable())                  -- {2, 6}
print(torch.linalg.vander(torch.tensor({1, 2, 3},
  {dtype = torch.float64})):totable())          -- increasing powers
print(torch.linalg.cross(torch.tensor({{1, 0, 0}}),
  torch.tensor({{0, 1, 0}})):totable())         -- {{0, 0, 1}}
print(torch.linalg.multi_dot({a, a}):totable()) -- optimal 2-matrix product
```

The remaining real-valued `torch.linalg` gaps are covered as well. `svdvals`
returns only descending singular values, `vecdot` reduces a dot product along
one broadcast dimension, `tensorinv` and `tensorsolve` group tensor dimensions
into matrix systems, and `householder_product` reconstructs `Q` from compact
reflectors. All five preserve autograd where the underlying operation does:

```lua
local s = a:svdvals()                      -- descending singular values
local d = torch.linalg.vecdot(x, y)        -- batched vector dot
local ai = torch.linalg.tensorinv(a, 2)    -- inverse of tensordot grouping
local x = torch.linalg.tensorsolve(A, b)   -- solve tensordot(A, x, x:dim()) = b
local q = torch.linalg.householder_product(A, tau)  -- Q from reflectors
```

The 0.29 linalg additions cover real symmetric eigendecomposition and
error-reporting factorizations. `eigh`/`eigvalsh` return ascending eigenvalues
and orthonormal eigenvectors, `cholesky_ex`/`inv_ex` return an int32 `info`
tensor instead of raising by default, and `torch.linalg.norm` dispatches
vector and matrix orders. `matrix_norm` now accepts batches, arbitrary
dimension pairs, `keepdim`, and the nuclear norm:

```lua
local a = torch.tensor({{2, 1}, {1, 3}}, {dtype = torch.float64})
local w, v = torch.linalg.eigh(a)            -- ascending w, columns of v
local l, info = torch.linalg.cholesky_ex(a)  -- info == 0 on success
local inv, iinfo = torch.linalg.inv_ex(a)    -- iinfo > 0 when singular
print(torch.linalg.norm(a, 2))               -- spectral norm
print(torch.linalg.matrix_norm(a, "nuc"))    -- nuclear norm
```

The 0.30 linalg additions cover error-reporting factorizations and symmetric
indefinite systems. `lu_factor_ex` returns the packed LU, 1-based pivots, and
an int32 `info`; `solve_ex` returns the solution and `info`; `ldl_factor` and
`ldl_factor_ex` expose compact LDL^T factors with 1-based block pivots; and
`ldl_solve` reuses them. The LU factor and solve forms carry autograd, while
the LDL factorization and solve are value-only on the CPU backend:

```lua
local a = torch.tensor({{4, 2}, {2, 3}}, {dtype = torch.float64})
local lu, lpiv, linfo = a:lu_factor_ex()
local ld, dpiv, dinfo = a:ldl_factor_ex()
local b = torch.tensor({{1}, {2}}, {dtype = torch.float64})
print(linfo:item(), dinfo:item())             -- 0, 0
print(lu:lu_solve(lpiv, b):totable())         -- {{-0.125}, {0.75}}
print(torch.linalg.ldl_solve(ld, dpiv, b):totable())  -- same solution
```

## Profiling

`torch.profiler` optionally records per-tensor-method call counts and CPU time,
without changing kernels:

```lua
torch.profiler.start()
local y = model:forward(x)
torch.profiler.stop()
print(torch.profiler.format())
```

See [docs/profiler.md](docs/profiler.md) for scope and limitations.

## Status

> **0.33.0 alpha**, CPU. Tensor core, autograd, I/O, luafun, `nn`/`optim`,
> data loading, sorting/linalg, the fusion JIT, fused optimizer updates,
> first-order fused backward, fused reductions, and **542 tests**. The 0.33
> additions are `torch.tril`/`triu`, `torch.trace`, `torch.diag`, and the
> `left=false` (right-side) option on `torch.linalg.solve`, with broadcasting,
> autograd, tests, and docs. The 0.32
> additions are `torch.aminmax`, `torch.count_nonzero`, `torch.isclose` plus
> `equal_nan` on `allclose`, `torch.nan_to_num`, the
> `torch.linalg.matmul`/`mm`/`mv`/`outer` aliases, and NaN-payload-safe value
> reads, with tests and docs. The 0.31
> additions are `torch.linalg.cholesky_inverse`, `torch.cholesky_solve`,
> the `torch.linalg.lu_unpack` alias, batched/hermitian `matrix_rank`, and
> batched `cond` with all real-valued orders, with real-valued validation,
> autograd where the underlying ATen op supports it, tests, and docs. The 0.30
> additions are `torch.linalg.lu_factor_ex`, `torch.linalg.solve_ex`,
> `torch.linalg.ldl_factor`, `torch.linalg.ldl_factor_ex`, and
> `torch.linalg.ldl_solve`, with error-reporting `info` tensors, compact
> LDL^T factors, real-valued validation, tests, and docs. The 0.29
> additions are `torch.linalg.eigh`/`eigvalsh`, `torch.linalg.cholesky_ex`,
> `torch.linalg.inv_ex`, the `torch.linalg.norm` dispatcher, and batched
> `torch.linalg.matrix_norm` with dimension-pair/keepdim support and the
> nuclear norm, with real-valued validation, autograd, tests, and docs. The
> 0.28 additions are `torch.linalg.svdvals`, `torch.linalg.vecdot`,
> `torch.linalg.tensorinv`, `torch.linalg.tensorsolve`, and
> `torch.linalg.householder_product`, with real-valued validation, autograd,
> tests, and docs. The 0.27 additions are `torch.lu_unpack`,
> `torch.diagonal`/`torch.linalg.diagonal`,
> `torch.vander`/`torch.linalg.vander`, `torch.cross`/`torch.linalg.cross`,
> and `torch.linalg.multi_dot`, with real-valued validation, autograd, tests,
> and docs. The 0.26 additions are `torch.linalg.matrix_exp` with
> scaling-and-squaring,
> square/batched/empty matrices, real float32/float64 validation, autograd,
> tests, docs, and benchmark coverage. The 0.25 additions are
> `torch.linalg.pinv` with SVD-based rank-deficient, rectangular, batched,
> and empty-matrix pseudoinverses, `atol`/`rtol` cutoffs, real
> float32/float64 validation, autograd, tests, docs, and benchmark coverage.
> The 0.24 additions are `torch.linalg.lu_factor` and
> `torch.linalg.lu_solve` with reusable packed factors, 1-based pivots,
> validated real-valued solves, batched and adjoint solves, autograd, tests,
> docs, and benchmark coverage.
> The 0.23 additions are `torch.linalg.lu` with partial pivoting, rectangular
> and batched inputs, autograd through `L` and `U`, tests, docs, and benchmark
> coverage. The 0.22 additions are `torch.linalg.solve_triangular` with a
> required `upper` flag, batched and right-side solves, autograd, tests, docs,
> and benchmark coverage. The 0.21 additions are `torch.linalg.matrix_power`
> and `torch.trapezoid` with autograd, tests, docs, and benchmark coverage.
> The 0.20 additions are `torch.linalg.lstsq` with driver/rcond support,
> autograd and benchmark coverage, and the related docs guards. The 0.19
> additions are `torch.linalg.slogdet`, autograd coverage for einsum, float16,
> and the detached NaN/Inf predicates, docs guards, and benchmark entries. The
> 0.18 additions are `torch.isfinite`, zero-copy `float16` `from_blob` buffers,
> and einsum edge/validation coverage. The 0.17 additions are validated
> `torch.einsum`, direct `float16` construction and printing parity, and
> `torch.isnan`/`torch.isinf` predicates. The 0.11 additions are cooperative
> `DataLoader` workers with bounded prefetch, ordered error propagation, and
> explicit iterator cleanup, documented in [docs/data.md](docs/data.md). The
> 0.10 SIMD-friendly polynomial `exp`/`tanh`/`sigmoid` helpers and the 0.9
> multi-layer/bidirectional RNN and LSTM with dropout remain included. CUDA
> and linear/conv epilogue fusion remain next (see [ROADMAP.md](ROADMAP.md)).

## Quick start

```lua
local torch = require("torch")

local x = torch.randn(2, 3)
print(x)
-- tensor([[-0.2273,  0.5088, -1.1477],
--         [ 0.1655,  3.0343, -0.1068]])

local w = torch.tensor({1, 2, 3}, {requires_grad = true})
local loss = (w * w):sum()
loss:backward()
print(w:grad())                    -- tensor([2., 4., 6.])

print(torch.arange(10):filter(function(v) return v % 2 == 0 end))
```

Training is PyTorch-shaped, with Lua conventions:

```lua
local nn = require("torch.nn")
local optim = require("torch.optim")
local data = require("torch.utils.data")

local model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 3))
local criterion = nn.CrossEntropyLoss()
local optimizer = optim.Adam(model:parameters(), {lr = 0.01})
local loader = data.DataLoader(
  data.TensorDataset(inputs, labels), {batch_size = 64, shuffle = true})

for epoch = 1, 50 do
  for batch_inputs, batch_labels in loader:iter() do
    local loss = criterion(model(batch_inputs), batch_labels)
    optimizer:zero_grad()
    loss:backward()
    optimizer:step()
  end
end
```

Functional pipelines use **luafun** (no Moses):

```lua
local fun = require("fun")        -- == torch.fun
local squares = fun.totable(fun.iter({1, 2, 3, 4})
  :map(function(x) return x * x end)
  :filter(function(x) return x % 2 == 0 end))
-- {4, 16}

for value in torch.randn(1000000):iter() do
  -- iterate without materialising a Lua table
end
```

## Build

Requirements: Linux/macOS, CMake ≥ 3.18, Ninja or Make, a C++17 compiler,
LuaJIT 2.1 with headers, `curl`, `unzip`. A C compiler (`gcc`) is needed for the
fast fusion backend; without it, fusion falls back to LuaJIT loops.

```bash
scripts/fetch_libtorch.sh     # downloads ~170 MB into third_party/libtorch
scripts/build.sh              # builds build/libluatorch.so
scripts/test.sh               # 542 tests
scripts/bench.sh              # micro-benchmarks
scripts/repl.sh               # interactive REPL
```

Or `make build`, `make test`, `make bench`, `make repl`.

If libtorch is already installed: `Torch_DIR=/path/to/libtorch/cmake scripts/build.sh`.

Install:

```bash
make install                 # PREFIX=$HOME/.local by default
export LUA_PATH="$HOME/.local/share/lua/5.1/?.lua;$HOME/.local/share/lua/5.1/?/init.lua;;"
```

Or install the Lua tree with LuaRocks and point `LUATORCH_LIB` at the native
shim built above:

```bash
luarocks --lua-version=5.1 make luatorch-0.8.0-1.rockspec
export LUATORCH_LIB="$PWD/build/libluatorch.so"
```

### Reproduce the PyTorch comparison

```bash
uv venv .venv-bench --python 3.12
uv pip install --python .venv-bench/bin/python torch \
  --index-url https://download.pytorch.org/whl/cpu
./scripts/compare.sh --compile
```

The fusion C cache lives in `.luatorch-cache/` (override with
`LUATORCH_CACHE`). Set `LUATORCH_FUSE_BACKEND=lua` to force the portable
LuaJIT backend; `gcc` unavailable → automatic fallback.

## What is implemented

| Area | Covered |
| --- | --- |
| Creation | `tensor`, `zeros`, `ones`, `empty`, `full`, `rand`, `randn`, `randperm`, `eye`, `arange`, `linspace`, `*_like` |
| Dtypes | float32/64/16, int8/16/32/64, uint8, bool; Torch7 typed constructors |
| Math | `+ - * / ^ %`, broadcasting, ~25 unary ops, `matmul/mm/bmm/dot`, `mv`, `outer`, `addmm`, `baddbmm`, comparisons, `where`, masks, `isclose`, `nan_to_num` |
| Reductions | `sum`, `mean`, `prod`, `max`/`min`/`aminmax`, `count_nonzero`, `argmax`/`argmin`, `var`/`std`, `norm`, `logsumexp`, `all`/`any` |
| Sorting | `sort`, `topk`, `unique`, `nonzero`, `gather`, `scatter` |
| Linear algebra | `inverse`, `inv_ex`, `det`, `solve`, `solve_ex`, `solve_triangular`, `lu`, `lu_factor`, `lu_factor_ex`, `lu_solve`, `lu_unpack`, `ldl_factor`, `ldl_factor_ex`, `ldl_solve`, `pinv`, `matrix_exp`, `diagonal`, `tril`, `triu`, `trace`, `diag`, `vander`, `cross`, `vecdot`, `multi_dot`, `svd`, `svdvals`, `tensorinv`, `tensorsolve`, `householder_product`, `lstsq`, `cholesky`, `cholesky_inverse`, `cholesky_solve`, `cholesky_ex`, `eigh`, `eigvalsh`, `norm`, `matrix_norm`, `matrix_rank`, `cond`, `qr`, `torch.linalg.*` |
| Autograd | `requires_grad`, `backward`, `grad`, `detach`, `no_grad` |
| Neural nets | Linear, Conv2d, pooling, BatchNorm/LayerNorm, Embedding, Dropout, activations, losses, Sequential/ModuleList |
| Optimizers | SGD/Adam/AdamW/RMSprop/Adagrad fused update kernels, fused gradient clipping, LR schedulers |
| Data | TensorDataset, DataLoader, custom datasets, seeded shuffling, bounded cooperative prefetch, ordered errors |
| I/O | PyTorch-format single tensors; `nn.save`/`model:load` state dicts |
| Fusion | Elementwise tracing, C JIT, LuaJIT fallback, `fused:into`, fused first-order backward, fused sum/mean/logsumexp reductions, fused last-dim softmax/log_softmax and cross-entropy, fused MSE/L1 losses, compile cache |
| Functional | luafun bridge, `Tensor:iter/map/filter/reduce/apply` |

## How the fusion engine works

1. `torch.fuse(fn)` traces `fn` with symbolic tensors, building an elementwise
   expression tree.
2. The tree is emitted as a C translation unit and compiled with
   `gcc -O3 -march=native -fopenmp` (cached by source hash), or as a LuaJIT
   loop if C compilation is unavailable.
3. Tensor inputs are made contiguous, output is allocated (or supplied via
   `into`), and one kernel reads all inputs and writes the result once.
4. `torch.fuse.train(fn)` additionally differentiates the expression
   symbolically, emits a vectorised backward kernel, and runs both through a
   C++ `torch::autograd::Function`, so `loss:backward()` reaches the compiled
   kernel.
5. `torch.fuse.reduce(fn, "sum"|"mean"|"logsumexp")` emits the expression and
   the reduction into one loop; when inputs require grad, a matching backward
   kernel is compiled and run through a C++ reduction autograd Function.
6. Any unsupported operation or shape mismatch falls back to eager ATen;
   inference fusion and reductions also fall back when gradients are required.
   Correctness never depends on the tracer.

See [docs/fusion.md](docs/fusion.md) for the API, supported operations, cache
and fallback rules.

## Design choices

- **1-based Lua surface.** Dimensions, indices, sort/topk indices, gather/scatter
  indices, argmax and class labels are 1-based; ATen sees 0-based values behind
  the FFI boundary.
- **Libtorch is the kernel library, not the architecture.** Large GEMMs and
  convolutions use ATen; elementwise chains are fused by us.
- **In-place and `into` are first-class.** They avoid allocation and LuaJIT GC
  pressure. Out-of-place remains the default for readability.
- **External-memory-aware GC.** Tensor bytes are tracked; a full collection is
  forced around a 64 MB threshold so C++ storage does not pile up. This single
  change took 1M out-of-place adds from ~3 ms to ~0.85 ms.
- **No Moses.** luafun is vendored as a single MIT file.

## Documentation

- [docs/fusion.md](docs/fusion.md) — fusion API, codegen, benchmark methodology
- [docs/api.md](docs/api.md) — full Lua API
- [docs/nn.md](docs/nn.md) — modules, losses, optimizers, data loading
- [docs/data.md](docs/data.md) — DataLoader options, lifecycle, workers, limits
- [docs/design.md](docs/design.md) — architecture and internals
- [docs/torch7-compat.md](docs/torch7-compat.md) — Torch7 ⇄ luatorch, Moses → luafun
- [ROADMAP.md](ROADMAP.md)

## Honest limitations

- Fusion handles elementwise expressions, `sum`/`mean`/`logsumexp`
  reductions with first-order backward, `softmax`/`log_softmax` over arbitrary
  dimensions, and cross-entropy for class counts up to 4096. Arbitrary-dimension
  softmax moves the axis last and pays for a contiguous copy; the last-dimension
  path avoids it.
- `torch.fuse.train` supports **first-order** derivatives and requires the C
  backend; higher-order gradients and the LuaJIT-only fallback use eager
  autograd.
- Fusion requires equal shapes (broadcasting falls back).
- We beat eager PyTorch on chains and small ops; we tie on GEMM and slightly
  lose on single out-of-place ops. Those are architectural facts, not bugs.
- No CUDA and no built-in datasets.
- RNN/LSTM are single-layer and unidirectional.
- State dictionaries exchange through safetensors; there is no direct loader for
  PyTorch `.pt` archives.

## Credits

PyTorch/ATen (BSD), LuaJIT (MIT), luafun by Roman Tsisyk (MIT), and Torch7 for
the ergonomics and the fun. See [LICENSE](LICENSE).
