# luatorch API reference (0.25.0)

All examples assume:

```lua
local torch = require("torch")
```

## Conventions

- **Dimensions are 1-based.** `t:size(1)` is the first dimension, `t:sum(2)`
  reduces the second dimension, `t:select(-1, 1)` selects the first element of
  the last dimension. Dimension `0` is an error.
- **Indices are 1-based.** Positive indices start at 1, negative indices count
  from the end (`-1` is the last element). `get`, `set`, `select`, `narrow`,
  `slice`, `cat`, and `index_select` follow this rule.
- **`argmax`/`argmin` return 1-based indices**, unlike Python PyTorch.
- **Shapes** can be passed as varargs (`torch.zeros(2, 3)`) or as a table
  (`torch.zeros({2, 3})`). `-1` is accepted by `reshape`/`view` as an inferred
  dimension.
- **Options tables** use `{dtype = ..., device = "cpu", requires_grad = true}`.
  `dtype` accepts the constants below or their string names.
- **Mutating methods** end in `_` and return the same tensor object. Everything
  else is out-of-place.
- **Reductions with `dim = nil`** reduce over every dimension and return a
  **0-dim tensor**; use `:item()` to obtain a Lua number.
- **Errors** are raised as Lua errors (`pcall` catches them) with the original
  ATen message.

## Fusion

`torch.fuse(fn, options?)` traces a pure elementwise function and compiles it
into a single kernel:

```lua
local fused = torch.fuse(function(a, b, c)
  return (a * b + c):relu()
end, {backend = "auto"})        -- "auto" | "c" | "lua"

local out = fused(a, b, c)
local buffer = torch.empty(a:sizes(), {dtype = a:dtype()})
fused:into(buffer, a, b, c)     -- reusable output, no allocation
```

Supported: `+ - * / ^`, unary `-`, `abs`, `relu`, `sigmoid`, `tanh`, `exp`, `log`,
`sqrt`, `clamp`. Tensor inputs must be float32/float64 with the same shape;
scalar arguments are runtime values. Anything else (reductions, broadcasting,
autograd inputs, unsupported ops) falls back to eager ATen.

The default C backend requires `gcc`; it caches compiled kernels in
`.luatorch-cache/` (`LUATORCH_CACHE` overrides). The LuaJIT backend is always
available. `fused.last_error` / `fused.last_c_error` explain fallbacks. See
[docs/fusion.md](fusion.md) for internals and benchmarks.

`torch.fuse.train(fn)` (alias `torch.fuse_train`) is the trainable variant. It
compiles a backward kernel and wires both directions into PyTorch's autograd
engine:

```lua
local train_fused = torch.fuse.train(function(a, b, c)
  return (a * b + c):relu()
end)

local output = train_fused(a, b, c)
output:sum():backward()         -- compiled backward kernel
```

It supports first-order derivatives of the elementwise operations above and
requires the C backend. Unsupported expressions, missing derivative rules, or
the absence of `gcc` fall back to eager autograd. `train_fused.last_error`
explains the fallback.

`torch.fuse.reduce(fn, reduction)` (alias `torch.fuse_reduce`) fuses an
elementwise expression directly into a scalar reduction, avoiding the
intermediate tensor. Supported reductions are `"sum"`, `"mean"` and
`"logsumexp"`:

```lua
local mse = torch.fuse.reduce(function(prediction, target)
  return (prediction - target) ^ 2
end, "mean")

local loss = mse(prediction, target)   -- 0-dim tensor
```

When inputs require grad, a compiled backward kernel runs through a C++
autograd Function, so the 0-dim loss can be
used normally in training. Unsupported expressions or a missing C compiler fall
back to eager expression + reduction. `reduce.last_error` explains the
fallback.

## Dtypes

| Constant | String | Bytes |
| --- | --- | --- |
| `torch.float32` / `torch.float` | `"float32"` | 4 |
| `torch.float64` / `torch.double` | `"float64"` | 8 |
| `torch.float16` / `torch.half` | `"float16"` | 2 |
| `torch.int64` / `torch.long` | `"int64"` | 8 |
| `torch.int32` / `torch.int` | `"int32"` | 4 |
| `torch.int16` / `torch.short` | `"int16"` | 2 |
| `torch.int8` / `torch.char` | `"int8"` | 1 |
| `torch.uint8` / `torch.byte` | `"uint8"` | 1 |
| `torch.bool` / `torch.boolean` | `"bool"` | 1 |

Lua numbers have no int/float distinction, so numeric tables and scalars infer
**float32**. Integer tensors are explicit:

```lua
torch.tensor({1, 2, 3}, {dtype = torch.int64})
torch.LongTensor({1, 2, 3})
```

`float16` values can be created directly from Lua tables and scalars (including
`torch.HalfTensor{...}`). Construction stages a float32 tensor and lets ATen
perform the conversion, so values use IEEE round-to-nearest-even and overflow
becomes `inf`:

```lua
torch.tensor({1.5, 2.5}, {dtype = torch.float16})
torch.tensor(1.5, {dtype = torch.float16})
torch.HalfTensor({1.5, -2.25})
```

Printing includes the `dtype=torch.float16` suffix and renders `inf`/`-inf`/
`nan` like the other floating dtypes. `torch.from_blob` also accepts raw
IEEE-754 binary16 elements, so a `uint16_t` buffer can back a `float16` tensor
without a copy.

## Creation

```lua
torch.tensor(value, opts?)          -- table/number/boolean/Tensor
torch.zeros(shape..., opts?)
torch.ones(shape..., opts?)
torch.empty(shape..., opts?)
torch.rand(shape..., opts?)
torch.randn(shape..., opts?)
torch.full(shape..., value, opts?)
torch.eye(n, opts?)
torch.arange(stop, opts?)           -- or (start, stop[, step], opts?)
torch.linspace(start, stop, steps, opts?)
torch.randperm(n, opts?)
```

`torch.tensor` infers `bool` for boolean tables and `float32` otherwise. Passing
an existing tensor returns a clone.

Like-constructors:

```lua
torch.zeros_like(t, opts?)
torch.ones_like(t, opts?)
torch.full_like(t, value, opts?)
torch.rand_like(t)
torch.randn_like(t)
```

### `torch.from_blob(buffer, shape[, opts])`

Creates a tensor sharing memory with a Lua string or LuaJIT FFI buffer. The
tensor keeps the buffer alive through `tensor._owner`. A Lua string is immutable,
so tensors backed by strings must be treated as read-only. FFI array buffers
are mutable and changes are visible in both directions.

```lua
local ffi = require("ffi")
local buffer = ffi.new("float[4]", { 1, 2, 3, 4 })
local t = torch.from_blob(buffer, { 2, 2 }, { dtype = "float32" })
t:set(1, 1, 9)       -- buffer[0] is now 9
```

`shape` is a table of dimensions. `opts.dtype` defaults to `float32`.
Array cdata sizes are checked with `ffi.sizeof`; pointer cdata must provide
`opts.nbytes`. `opts.requires_grad` is honored.

All dtypes are supported, including `float16`. Half-precision buffers contain
raw IEEE-754 binary16 elements (2 bytes each), for example a `uint16_t` array
of bit patterns; LuaJIT's `ffi.new("float[...]")` buffer only matches
`dtype = "float32"`. Use `:to(torch.float32)` to read half values as Lua
numbers:

```lua
local bits = ffi.new("uint16_t[4]", { 0x3C00, 0x4000, 0x4200, 0x3800 })
local h = torch.from_blob(bits, { 2, 2 }, { dtype = torch.float16 })
print(h:to(torch.float32))  -- values 1, 2, 3, 0.5
bits[0] = 0x4000            -- h:get(1, 1) is now 2
```

### Torch7-style constructors

```lua
torch.Tensor(2, 3)                  -- uninitialised float32
torch.Tensor({1, 2, 3})             -- float32 values
torch.FloatTensor(2, 3)
torch.DoubleTensor({1.5, 2.5})
torch.LongTensor({1, 2, 3})
torch.IntTensor(...)   torch.ShortTensor(...)   torch.CharTensor(...)
torch.ByteTensor(...)  torch.BoolTensor(...)    torch.HalfTensor(...)
```

## Tensor metadata

| Method | Description |
| --- | --- |
| `t:dim()` / `t:ndimension()` | number of dimensions |
| `t:numel()` | number of elements |
| `t:size(dim?)` | one dimension, or a table of all sizes |
| `t:sizes()` / `t:shape()` | table of sizes |
| `t:dtype()` | dtype string |
| `t:dtype_id()` | numeric dtype id |
| `t:device()` | `"cpu"` (CUDA is not built yet) |
| `t:is_cuda()` | `false` |
| `t:is_contiguous()` | whether the memory layout is contiguous |
| `t:is_floating_point()` | true for float16/32/64 |
| `t:requires_grad()` | autograd flag |
| `t:element_size()` / `t:nbytes()` | element and total byte size |
| `t:type()` | Torch7 type name, e.g. `"torch.FloatTensor"` |

## Dtype conversion and copies

```lua
t:to(dtype_or_device)   -- new tensor; also accepts "cpu"
t:type_as(other)
t:float()  t:double()  t:long()  t:int()  t:half()  t:bool()
t:clone()  t:contiguous()  t:detach()
t:copy_(src)            -- in-place copy with broadcasting
```

## Scalar access and indexing

```lua
t:get(i, j, ...)        -- Lua number / boolean
t:set(i, j, ..., value)
t:item()                -- the single element of a 1-element tensor
t[index]                -- selects dimension 1; t[1] == t:select(1, 1)
t[index] = number       -- 1-d scalar assignment
t[index] = tensor       -- copy into the selected slice
t[mask]                 -- boolean mask indexing -> 1-d tensor
```

Shape-preserving selection:

```lua
t:select(dim, index)
t:narrow(dim, start, length)
t:slice(dim, start, stop, step?)   -- start/stop are inclusive, 1-based
t:flip(dim)
t:masked_select(mask)
t:index_select(dim, indices)       -- indices is an integer tensor, 1-based
t:masked_fill(mask, value)
t:gather(dim, index)               -- 1-based index tensor
t:index_select_batch(dim, {i1,i2}) -- one C call for several index tensors
t:scatter(dim, index, source)      -- out-of-place, 1-based index tensor
t:nonzero()                        -- 1-based coordinates, shape (N, ndim)
t:sort(dim?, descending?)          -- values, 1-based indices
t:topk(k, dim?, largest?, sorted?) -- values, 1-based indices
t:take_along_dim(indices, dim?)    -- gather along a dimension
t:unique(sorted?, inverse?, counts?) -- values[, inverse, counts]; inverse is 1-based
t:median(dim?)                    -- median value or dimension median
t:quantile(q, dim?, keepdim?)     -- interpolated quantile
```

Examples:

```lua
local m = torch.arange(1, 13):reshape(3, 4)  -- 3x4, values 1..12
m:select(1, 2)          -- tensor([5, 6, 7, 8])
m:slice(2, 2, 3)        -- 3x2, second and third columns
m:slice(1, -2, -1)      -- last two rows
m[m:gt(6)]              -- tensor([7, 8, 9, 10, 11, 12])
```

## Elementwise math

Two-tensor and scalar variants:

```lua
t:add(x)  t:sub(x)  t:mul(x)  t:div(x)  t:pow(x)  t:remainder(x)
t:add_(x) ...                       -- in-place
t:maximum(other)  t:minimum(other)  -- tensors only
```

Operators: `+ - * / ^` and unary `-`, including scalar-left forms
(`2 - t`, `1 / t`, `2 ^ t`). `cmul`/`cdiv` are Torch7 aliases for
`mul`/`div` (and `cmul_`/`cdiv_`).

Comparisons return Boolean tensors: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
(tensor or scalar argument). Logical: `logical_and`, `logical_or`,
`logical_not`.

Unary methods (each has an `_` in-place version):

`neg abs exp log log1p sqrt rsqrt sin cos tan asin acos atan sinh cosh tanh
sigmoid relu floor ceil round trunc sign square reciprocal`

Plus:

```lua
t:clamp(min, max)     t:clamp_(min, max)
t:fill_(value)        t:zero_()
```

## Linear algebra

```lua
a:matmul(b)     -- batched matmul with broadcasting
a:mm(b)         -- 2-D matrix multiplication
a:bmm(b)        -- batched 3-D matmul
a:dot(b)        -- 1-D dot product
a:mv(v)         -- matrix-vector product
a:outer(v)      -- outer product
a:addmm(m1, m2, beta?, alpha?)    -- beta*a + alpha*(m1 @ m2)
a:baddbmm(m1, m2, beta?, alpha?)  -- batched addmm
```

### Linear algebra

```lua
a:inverse()
a:det()
a:slogdet()         -- returns sign, logabsdet; batched and differentiable
a:solve(b)          -- solve a x = b (b may be a vector or matrix)
a:solve_triangular(b, opts)  -- triangular solve; opts.upper is required
a:lstsq(b, opts?)   -- least squares; returns x, residuals, rank, singular values
a:cholesky(upper?)  -- A = L L^T (or U^T U)
a:svd(full_matrices?)  -- returns U, S, Vh
a:qr(mode?)            -- returns Q, R; mode defaults to "reduced"
a:lu()                 -- pivoted LU; returns P, L, U with A = P L U
a:lu_factor()          -- packed LU and 1-based pivots; raises if singular
a:lu_solve(pivots, b, opts?)  -- solve from a packed factorization
a:pinv(opts?)          -- Moore-Penrose pseudoinverse; opts.atol/rtol
a:matrix_power(n)      -- integer power; negative n inverts (floating only)

torch.linalg.inv(a)      torch.linalg.det(a)      torch.linalg.slogdet(a)
torch.linalg.solve(a, b) torch.linalg.cholesky(a) torch.linalg.svd(a)
torch.linalg.solve_triangular(a, b, opts)
torch.linalg.qr(a)       torch.linalg.lstsq(a, b, opts?)  -- aliases
torch.linalg.lu(a)       torch.linalg.lu_factor(a)
torch.linalg.lu_solve(lu, pivots, b, opts?)  torch.linalg.matrix_power(a, n)
torch.linalg.pinv(a, opts?)
```

`slogdet` computes the signed log-determinant without forming the determinant,
so it stays finite for very large or small magnitudes. `sign` is `-1`, `0`, or
`1`; `logabsdet` is real. For a square matrix `exp(logabsdet) * sign == det`,
and gradients flow through `logabsdet` as `A^{-T}`. Batched inputs return
`sign` and `logabsdet` with the batch dimensions; square matrices are required.

`solve_triangular` solves `A X = B` when `left` is true (the default) and
`X A = B` when it is false, assuming `A` is triangular. It requires an options
table with `upper` set to a boolean, so a lower/upper mix-up cannot silently
produce a wrong answer; `left` defaults to true and `unitriangular` defaults to
false, in which case the stored diagonal is treated as ones. For a left solve,
`B` may be 1-D of length `n` or have shape `(*, n, k)`; for a right solve it
must be at least 2-D with shape `(*, k, n)`. Batch dimensions broadcast. Only
float32 and float64 are accepted because CPU LAPACK has no half-precision
triangular solve, and both operands must share the dtype. The solution is
differentiable with respect to both `A` and `B`, and gradients with respect to
the unused triangle are zero.

```lua
local a = torch.tensor({{2, 0}, {1, 3}}, {dtype = torch.float64})
local b = torch.tensor({2, 4}, {dtype = torch.float64})
print(a:solve_triangular(b, {upper = false}):totable())  -- {1, 1}
```

`lu` computes a pivoted LU factorization `A = P L U` with partial pivoting.
For an input of shape `(*, m, n)`, it returns `P` of shape `(*, m, m)`, `L`
of shape `(*, m, k)`, and `U` of shape `(*, k, n)`, where `k = min(m, n)`.
`P` is a permutation matrix and is detached from autograd; `L` is unit lower
triangular, `U` is upper triangular, and both are differentiable. Square,
rectangular, and batched matrices are supported. Only float32 and float64 are
accepted, because CPU LAPACK does not implement other dtypes, and the
unpivoted variant is not exposed:

```lua
local a = torch.tensor({{0, 1}, {1, 0}}, {dtype = torch.float64})
local p, l, u = a:lu()
print(p:matmul(l):matmul(u):totable())  -- {{0, 1}, {1, 0}}
```

`lu_factor` returns the compact factorization `(LU, pivots)` used by
`lu_solve`. `LU` has the same shape as `A`: `U` occupies the diagonal and
upper triangle, and the strict lower triangle stores the multipliers of `L`.
`pivots` is an int32 tensor of 1-based row interchanges with shape
`(*, min(m, n))` for an input of shape `(*, m, n)`. Singular matrices raise
instead of returning a factorization that would divide by zero later.
Rectangular and batched inputs are supported, but `lu_solve` requires a
square factor.

`lu_solve` solves `A X = B` from the `(LU, pivots)` pair. `left = false`
solves `X A = B`, and `adjoint = true` solves `A^T X = B` (the real-valued
transpose). `B` must be at least 2-D: for a left solve it has shape
`(*, n, k)` with `n` equal to the factor size, and for a right solve its
last dimension must equal the factor size. A single right-hand side is a
column matrix. The pivot batch shape must match the factor exactly, while
`B`'s batch dimensions may broadcast. `LU` and `B` must share the
float32/float64 dtype. The solution is differentiable with respect to `LU`
and `B`; the integer pivots stay detached.

```lua
local a = torch.tensor({{4, 3}, {2, 1}}, {dtype = torch.float64})
local lu, pivots = a:lu_factor()
local b = torch.tensor({{1}, {2}}, {dtype = torch.float64})
print(lu:lu_solve(pivots, b):totable())  -- {{2.5}, {-3}}
```

`pinv` computes the Moore-Penrose pseudoinverse through ATen's SVD-based
`linalg_pinv`. Unlike `inverse`, it accepts rectangular, rank-deficient,
batched, and empty matrices: for an input of shape `(*, m, n)` the result has
shape `(*, n, m)` and satisfies the four Moore-Penrose identities. Singular
values at or below `max(atol, sigma_1 * rtol)` are treated as zero, where
`sigma_1` is the largest singular value. When `rtol` is omitted it defaults to
`max(m, n) * eps`; supplying a positive `atol` without `rtol` sets `rtol` to
zero, matching PyTorch:

```lua
local a = torch.tensor({{1, 2}, {3, 4}, {5, 7}}, {dtype = torch.float64})
print(a:pinv():sizes())                  -- {2, 3}
print(a:pinv({rtol = 1e-6}):sizes())     -- {2, 3}
```

Only real float32/float64 inputs are accepted because the CPU LAPACK SVD path
has no half-precision kernel and luatorch has no complex dtype; float16,
integer, and bool inputs are rejected with Lua errors. `atol` and `rtol` must
be non-negative numbers. The pseudoinverse is differentiable with respect to
the input, but its derivative diverges at repeated singular values, so
gradients near a rank change can be ill-conditioned.

`lstsq` minimizes `||A x - b||` and returns four values: `solution`,
`residuals`, `rank`, and `singular_values`. For a matrix `A` of shape
`(*, m, n)`, `b` may be `(*, m)` (a vector of right-hand sides) or
`(*, m, k)`; the solution is then `(*, n)` or `(*, n, k)`. `residuals` holds
the per-column sum of squared residuals (`(*, k)`), `rank` is an int64 tensor
of shape `(*)`, and `singular_values` has shape `(*, min(m, n))`; the last
three are empty when the chosen driver does not compute them. Options:

```lua
local x, residuals, rank, singular = a:lstsq(b, {
  driver = "gelsd",   -- "gels", "gelsy" (ATen CPU default), "gelsd", "gelss"
  rcond = 1e-10,      -- singular values below rcond * largest are treated as 0
})
```

`"gels"` assumes full rank and returns empty `rank`/`singular_values`;
`"gelsy"` returns empty `singular_values` and empty residuals. Residuals are
also empty for underdetermined or rank-deficient systems. `rcond` must be
non-negative when supplied; it is honored by `gelsy`, `gelsd`, and `gelss`,
and `rcond = 0` uses machine precision. The solution is differentiable. CPU
LAPACK only accepts float32 and float64, so cast float16 inputs first.

`matrix_power` raises a square matrix to an integer power with binary
exponentiation and supports batches of square matrices. Non-negative
exponents accept floating and integer dtypes; negative exponents call
`inverse`, so they require a floating-point tensor. `n = 0` returns a fresh
identity, and `n = 1` returns a clone rather than aliasing the input. The
result is differentiable for floating-point inputs, including negative
exponents. Float16 works for non-negative exponents through `matmul`; negative
powers follow the usual CPU LAPACK float32/float64 restriction.

### Einstein summation

`torch.einsum(equation, ...)` computes arbitrary products and reductions:

```lua
torch.einsum("ij,jk->ik", a, b)             -- matrix multiply
torch.einsum("ij,jk", a, b)                 -- implicit output
torch.einsum("...ij,...jk->...ik", a, b)    -- batched matmul
torch.einsum("ij->ji", a)                   -- transpose
torch.einsum("ii->i", a)                    -- diagonal
torch.einsum("ij->", a)                     -- sum every element
torch.einsum("i,->i", v, scalar)            -- scalar operand
```

Subscripts are ASCII letters; `...` broadcasts batch dimensions; whitespace is
ignored. Empty subscripts denote 0-dim operands, so `",->"` multiplies two
scalars and a label repeated within one operand takes its diagonal (`"ii->i"`,
`"...ii->..."`). An output may contain `...` even when no input does; it then
covers zero dimensions. The Lua layer validates the equation grammar, operand
count, ellipsis placement, and output labels before calling ATen. Rank and
shape mismatches are reported by ATen, and gradients flow through einsum like
any composite operation. An implicit output (no `->`) keeps labels that appear
exactly once across the inputs, sorted alphabetically, and sums over labels
that repeat -- matching PyTorch.

All operands must share exactly one dtype; mixed dtypes are rejected instead of
implicitly promoted, matching PyTorch's einsum. Boolean operands are rejected
because ATen has no boolean einsum kernels -- cast them to an integer or
floating dtype first.

## Neural-network primitives

These back `torch.nn`; they are also useful when writing custom layers.

```lua
t:softmax(dim?)        t:log_softmax(dim?)     -- any dim; fused via layout move
t:uniform_(low?, high?)                       -- random in-place init
t:normal_(mean?, std?)      t:exponential_(lambda?)
t:bernoulli_(p?)            t:multinomial(n, replacement?)  -- 1-based samples

torch.softmax(t, dim)  torch.log_softmax(t, dim)
torch.conv2d(input, weight, bias, stride?, padding?, dilation?, groups?)
torch.max_pool2d(input, kernel, stride?, padding?, dilation?, ceil_mode?)
torch.avg_pool2d(input, kernel, stride?, padding?, ceil_mode?, count_include_pad?)
torch.embedding(weight, indices)                -- 1-based indices
torch.layer_norm(input, normalized_shape, weight?, bias?, eps?)
torch.cross_entropy(input, target, reduction?)  -- targets are 1-based
torch.cross_entropy(input, target, { weight = w, ignore_index = i,
  label_smoothing = s })                         -- Lua option path
torch.nll_loss(log_probs, target, reduction?)
torch.binary_cross_entropy_with_logits(input, target, reduction?)
torch.uniform_(t, low, high)
```

`reduction` is `"none"`, `"mean"` (default) or `"sum"`.

## Data loading

`DataLoader` options, iterator lifecycle, worker semantics, and limits are
documented in [docs/data.md](data.md).

```lua
local data = require("torch.utils.data")   -- or torch.utils.data

local dataset = data.TensorDataset(inputs, labels)  -- 1-based get()
local loader = data.DataLoader(dataset, {
  batch_size = 64,
  shuffle = true,
  drop_last = false,
  num_workers = 2,      -- cooperative coroutine workers (default 0)
  prefetch = 4,         -- batches prepared ahead (default 2*num_workers, or 0)
  seed = 42,            -- optional deterministic shuffle
})
for batch_inputs, batch_labels in loader:iter() do
  -- ...
end
```

Custom datasets implement `size()` and `get(index)`; an optional `batch(indices)`
can return a whole batch at once. The dataset contract is unchanged by workers
and prefetch.

`loader:iter()` returns a callable iterator. It is exhausted with `nil`, closes
its workers at the end of the epoch, and can be closed early with
`iterator:close()` (or `loader:close()` for every live iterator). Iterators that
are abandoned without closing are finalized by garbage collection.

### Worker and prefetch model

`num_workers` creates Lua coroutine workers in the current Lua state. These are
**cooperative, not OS threads or processes**: only one Lua coroutine runs at a
time, and a worker is resumed only while the loader is preparing batches. A
worker yields one complete batch per resume, so dataset access remains
single-threaded and needs no locking. This avoids thread/process start-up,
copied Lua states, and IPC, but it also means workers cannot run in parallel
with each other or with training math. In particular, a long-running ATen call
does not yield to Lua, so it does not overlap with prefetching work.

`prefetch` bounds how many batches may be prepared ahead of the batch currently
being consumed. The queue holds at most `prefetch + 1` batches; `0` computes each
batch only when requested. Workers and prefetch change neither the shuffle nor
the batch order: batches are always delivered in the same deterministic order
for a given seed and dataset size.

Without `seed`, shuffling uses the global `torch` RNG (`torch.manual_seed`).
With `seed`, the loader uses a private deterministic PRNG, does not consume the
global RNG, and produces the same order each epoch. Worker failures are re-raised
on the batch that failed, with worker context, and close the iterator. See
[docs/nn.md](nn.md#data-loading) for datasets, modules, losses and optimizers.

## Shape manipulation

```lua
t:reshape(2, 3)         t:reshape({2, 3})      t:reshape(-1)
t:view(2, 3)
t:transpose(dim1, dim2)
t:t()
t:permute(dim1, dim2, ...)
t:squeeze(dim?)         -- no dim: remove all size-1 dims
t:unsqueeze(dim)
t:flatten(start_dim?, end_dim?)
t:expand(sizes...)
torch.cat({t1, t2}, dim?)      -- default dim = 1
torch.stack({t1, t2}, dim?)    -- default dim = 1
```

`expand` follows PyTorch broadcasting rules (including `-1` to keep a size).

## Reductions

```lua
t:sum(dim?, keepdim?)
t:mean(dim?, keepdim?)
t:prod(dim?, keepdim?)
t:max(dim?, keepdim?)          -- values[, indices] when dim is given
t:min(dim?, keepdim?)          -- values[, indices] when dim is given
t:argmax(dim?, keepdim?)       -- 1-based indices
t:argmin(dim?, keepdim?)
t:var(dim?, unbiased?, keepdim?)
t:std(dim?, unbiased?, keepdim?)
t:norm(p?, dim?, keepdim?)     -- p defaults to 2; math.huge supported
t:logsumexp(dim?, keepdim?)
t:cumsum(dim?)                 -- dim defaults to 1
t:cumprod(dim?)
t:trapezoid(x?, opts?)         -- integrate along a dimension
t:all(dim?, keepdim?)
t:any(dim?, keepdim?)
```

`max`/`min` return two values when `dim` is provided:

```lua
local values, indices = t:max(2)
```

`trapezoid` applies the trapezoidal rule along one dimension and removes that
dimension from the result. `x` supplies the sample positions either as a 1-D
tensor with one value per sample or as a tensor shaped like `t`; when `x` is
omitted, `opts.dx` sets a uniform spacing (default `1`). `opts.dim` is 1-based
and defaults to the last dimension. Exactly one of `x` and `dx` may be given:

```lua
local y = torch.tensor({1, 2, 3}, {dtype = torch.float64})
torch.trapezoid(y)                              -- 4.0
torch.trapezoid(y, nil, {dx = 0.5})             -- 2.0
torch.trapezoid(y, torch.tensor({0, 1, 3},
  {dtype = torch.float64}))                     -- 6.5
```

The input must be floating point; integer and bool tensors are rejected
instead of being promoted (cast them first, unlike PyTorch). Integral `x` is
cast to the input dtype, but a floating `x` must already match it. The result
is differentiable with respect to both the values and a floating-point `x`. A
dimension with fewer than two samples integrates to zero.

## Autograd

```lua
local x = torch.tensor({1, 2, 3}, {requires_grad = true})
local loss = (x * x):sum()
loss:backward()
print(x:grad())                 -- tensor([2., 4., 6.])

x:zero_grad()
x:requires_grad_(true)
x:has_grad()
x:grad()                        -- nil before the first backward
x:detach()
x:backward(gradient?)           -- explicit upstream gradient
```

Global grad mode:

```lua
torch.is_grad_enabled()
torch.set_grad_enabled(false)
local y = torch.no_grad(function()
  return (x * 2):sum()          -- does not build a graph
end)
```

Gradients keep the tensor dtype: a `float16` graph produces `float16`
gradients, and casts promote the gradient back to the source dtype. `float16`
autograd covers elementwise math, reductions, matmul and einsum, `nn.Linear`,
the losses, softmax/log-softmax, and `where`/masked updates. CPU LAPACK
operations (`inverse`, `det`, `slogdet`, `svd`, `pinv`, `cholesky`, `qr`,
`lstsq`) reject `float16` through ATen, and the custom optimizer kernels
require float32 or float64 parameters; cast to float32 for those paths.

`torch.einsum` is differentiable for every equation form -- implicit outputs,
diagonals and traces, ellipses/batches, and scalar operands -- matching ATen's
composite backward. `torch.isnan`, `torch.isinf`, and `torch.isfinite` return
detached bool tensors that cannot be differentiated; use them to build masks
for `where`, `masked_select`, or boolean indexing, where gradients flow only
through the selected values.

`matrix_power` and `trapezoid` are composite operations, so their graphs are
built from differentiable ATen calls: positive matrix powers differentiate
through `matmul` (negative powers also through `inverse`), and `trapezoid`
differentiates with respect to both the values and a floating-point `x`.

## Functional utilities and luafun

```lua
t:iter()                  -- iterator over elements (contiguous order)
t:map(fn)                 -- new tensor, same shape and dtype
t:filter(pred)            -- 1-d tensor of matching values
t:reduce(fn, initial?)    -- fold
t:apply(fn)               -- in-place map
t:fun()                   -- fun.iter(t:totable())
t:totable()               -- nested Lua table (or a scalar for 0-dim)
```

`torch.fun` (alias `torch.luafun`) is the vendored
[luafun](https://github.com/rtsisyk/luafun) module:

```lua
local fun = torch.fun
local total = fun.iter({1, 2, 3, 4}):reduce(function(a, b) return a + b end, 0)
local evens = fun.totable(fun.iter({1, 2, 3, 4}):filter(function(x) return x % 2 == 0 end))
```

## Comparison helpers

```lua
a:equal(b)              -- same shape and exactly equal values
a:allclose(b, rtol?, atol?)
torch.equal(a, b)
torch.allclose(a, b, rtol?, atol?)
```

## NaN and Inf predicates

```lua
torch.isnan(t)          -- bool tensor, true where the value is NaN
torch.isinf(t)          -- bool tensor, true where the value is +/-Inf
torch.isfinite(t)       -- bool tensor, true where neither NaN nor +/-Inf
                        -- integer and bool inputs are always finite
t:isnan()               -- method aliases
t:isinf()
t:isfinite()
```

The result has the same shape as the input and never participates in autograd:
predicates return a detached bool tensor even when the input requires grad.
Float16/32/64 values are checked; integer and bool tensors produce all-true
masks, like PyTorch.

## Serialization

```lua
t:save("tensor.pt")
torch.save(t, "tensor.pt")
local copy = torch.load("tensor.pt")
torch.print(t)          -- prints t:format()
```

Files use the PyTorch zip archive format (`torch::save` in C++), so tensors can
be exchanged with C++/Python tooling that understands that format.

Module state dictionaries can be exchanged as safetensors:

```lua
model:save_safetensors("model.safetensors")
model:load_safetensors("model.safetensors")
model:load_safetensors("model.safetensors", { strip_prefix = "model." })
```


## Module-level aliases

`torch.add`, `sub`, `mul`, `div`, `pow`, `matmul`, `mm`, `bmm`, `dot`, `sum`,
`mean`, `prod`, `max`, `min`, `argmax`, `argmin`, `transpose`, `reshape`,
`clamp`, `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt`, `abs` are thin
wrappers around the corresponding tensor methods.

Other module helpers:

```lua
torch.is_tensor(x)
torch.manual_seed(seed)
torch.set_default_dtype(dtype)
torch.get_default_dtype()
torch.dtype_name(dtype)
torch.get_num_threads() / torch.set_num_threads(n)
torch.sort / topk / unique / median / quantile / bincount / histc / nonzero / gather / scatter / multinomial
torch.ger(a, b)                  -- outer product of two vectors
torch.einsum(equation, ...)      -- validated Einstein summation
torch.trapezoid(y, x?, opts?)    -- trapezoidal integration along one dim
torch.isnan(t) / torch.isinf(t) / torch.isfinite(t)  -- bool predicates
torch.linalg.matrix_norm(a, ord) -- fro, 2, -2, 1, -1, inf, -inf
torch.linalg.cond(a, p?)         -- 2, 1, inf, fro
torch.linalg.matrix_rank(a, tol?)
torch.linalg.matrix_power(a, n)  -- integer matrix power
torch.linalg.vector_norm(a, ord?, dim?, keepdim?)
torch.linalg.slogdet(a)          -- returns sign, logabsdet
torch.linalg.lstsq(a, b, opts?)  -- returns x, residuals, rank, singular values
torch.linalg.pinv(a, opts?)      -- Moore-Penrose pseudoinverse
torch.mv / outer / addmm / baddbmm
torch.ALL_DIMS            -- -1000 sentinel (internal, exposed for clarity)
torch.profiler            -- opt-in tensor-method profiler (see docs/profiler.md)
torch.VERSION             -- luatorch version string ("0.25.0")
torch.version             -- libtorch version string
```
