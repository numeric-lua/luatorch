# luatorch API reference (0.34.0)

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
t:argsort(dim?, descending?)       -- 1-based sort indices only
t:topk(k, dim?, largest?, sorted?) -- values, 1-based indices
t:take_along_dim(indices, dim?)    -- gather along a dimension
t:take(index)                      -- flattened 1-based gather; index shape is output
t:unique(sorted?, inverse?, counts?) -- values[, inverse, counts]; inverse is 1-based
t:median(dim?)                    -- median value or dimension median
t:quantile(q, dim?, keepdim?)     -- interpolated quantile
t:diff(n?, dim?)                  -- n-th discrete difference, default n=1, last dim
```

Examples:

```lua
local m = torch.arange(1, 13):reshape(3, 4)  -- 3x4, values 1..12
m:select(1, 2)          -- tensor([5, 6, 7, 8])
m:slice(2, 2, 3)        -- 3x2, second and third columns
m:slice(1, -2, -1)      -- last two rows
m[m:gt(6)]              -- tensor([7, 8, 9, 10, 11, 12])
```

`argsort` returns only the 1-based sort indices, with the same default
dimension (the last one) and `descending` flag as `sort`. `diff(n?, dim?)`
computes the `n`-th discrete difference along the last dimension by default;
`n = 0` returns a clone, empty or length-1 dimensions produce an empty result,
and subtraction carries autograd.

`torch.searchsorted(sorted_sequence, values, right?)` binary-searches a 1-D
sorted sequence and returns int64 insertion positions. Positions are **1-based**
and range from 1 to `n + 1`; the default is the leftmost position, and `right`
selects after equal elements. `torch.bucketize(input, boundaries, right?)` is
the same search with the arguments reversed and accepts input of any shape.

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

`neg abs exp expm1 log log1p log2 log10 erf erfc sinc sqrt rsqrt sin cos tan
asin acos atan sinh cosh tanh sigmoid relu floor ceil round trunc sign square
reciprocal`

`log2`/`log10` are base-2/base-10 logarithms, `expm1` is `exp(x) - 1` with
preserved precision near zero, `erf`/`erfc` are the error and complementary
error functions, and `sinc` is the normalized `sin(pi x)/(pi x)` with
`sinc(0) = 1`. All are ATen-backed and differentiable, and the module-level
forms are `torch.log2`, `torch.log10`, `torch.expm1`, `torch.erf`,
`torch.erfc`, and `torch.sinc`.

Plus:

```lua
t:clamp(min, max)     t:clamp_(min, max)
t:addcmul(t1, t2, value?)   -- self + value * (t1 * t2)
t:addcdiv(t1, t2, value?)   -- self + value * (t1 / t2)
t:lerp(end, weight)         -- self + weight * (end - self)
t:logaddexp(other)          -- log(exp(self) + exp(other))
t:logaddexp2(other)         -- log2(2^self + 2^other)
t:hypot(other)              -- sqrt(self^2 + other^2)
t:frac()                    -- self - trunc(self)
t:softplus(beta?, threshold?)  -- log(1 + exp(beta*x)) / beta
t:elu(alpha?)               -- x if x > 0, alpha*(exp(x)-1) otherwise
t:selu()                    -- scaled elu
t:silu()                    -- x * sigmoid(x)
t:gelu(approximate?)        -- "none" (erf) or "tanh" approximation
t:clamp_min(value)          -- max(self, value), floating dtypes
t:clamp_max(value)          -- min(self, value), floating dtypes
t:logit(eps?)               -- inverse sigmoid; eps clamps to [eps, 1-eps]
t:fill_(value)        t:zero_()
```

`addcmul` and `addcdiv` multiply or divide two tensors, scale the result by
`value` (default `1`), and add it to the input. Broadcasting follows the
ordinary elementwise rules, bool inputs are rejected, and autograd flows to
all operands. `lerp` interpolates between two floating-point tensors; `weight`
may be a number or a broadcastable tensor of the same dtype, and gradients flow
to both endpoints and a tensor weight. `torch.addcmul`, `torch.addcdiv`, and
`torch.lerp` are the module-level forms.

`logaddexp`, `logaddexp2`, and `hypot` take two broadcastable floating-point
tensors of the same dtype and are computed with max/magnitude shifts so large
finite inputs do not overflow. `hypot` returns zero for `hypot(0, 0)` and keeps
infinite inputs infinite. `frac` returns `self - trunc(self)`: the signed
fractional part for floating dtypes and zero for integer dtypes, with
autograd flowing through the subtraction. `torch.logaddexp`,
`torch.logaddexp2`, `torch.hypot`, and `torch.frac` are the module-level
forms.

The activation helpers accept float16/float32/float64 tensors and are
differentiable. `softplus(beta?, threshold?)` defaults to `beta = 1` and
`threshold = 20`; the stable max-shift form switches to the linear branch above
the threshold. `elu(alpha?)` defaults `alpha = 1`, `selu()` applies the fixed
`alpha`/`scale` constants, and `silu()` is `x * sigmoid(x)`. `gelu()` uses the
exact `erf` form by default and the standard tanh approximation with
`gelu("tanh")`. Module-level forms are `torch.softplus`, `torch.elu`,
`torch.selu`, `torch.silu`, and `torch.gelu`.

`clamp_min` and `clamp_max` are one-sided clamps with `_` in-place variants;
they accept float16/float32/float64 inputs and preserve NaN entries.
`logit(eps?)` is the inverse sigmoid `log(x/(1-x))`; with `eps` in `[0, 0.5)`
the input is clamped to `[eps, 1-eps]` first. Module-level forms are
`torch.clamp_min`, `torch.clamp_max`, and `torch.logit`.

## Linear algebra

```lua
a:matmul(b)     -- batched matmul with broadcasting
a:mm(b)         -- 2-D matrix multiplication
a:bmm(b)        -- batched 3-D matmul
a:dot(b)        -- 1-D dot product
a:mv(v)         -- matrix-vector product
a:outer(v)      -- outer product
a:kron(b)       -- Kronecker product, ranks aligned from the right
a:addmm(m1, m2, beta?, alpha?)    -- beta*a + alpha*(m1 @ m2)
a:baddbmm(m1, m2, beta?, alpha?)  -- batched addmm
```

### Linear algebra

```lua
a:inverse()
a:inv_ex(opts?)        -- inverse plus int32 info; opts.check_errors
a:tensorinv(ind?)   -- inverse of tensordot over the first ind dimensions
a:tensorsolve(b, dims?)  -- solve tensordot(a, x, x:dim()) = b
a:det()
a:slogdet()         -- returns sign, logabsdet; batched and differentiable
a:solve(b, opts?)   -- solve A x = b; opts.left=false solves x A = b
a:solve_ex(b, opts?) -- solve plus int32 info; opts.left/check_errors
a:solve_triangular(b, opts)  -- triangular solve; opts.upper is required
a:lstsq(b, opts?)   -- least squares; returns x, residuals, rank, singular values
a:cholesky(upper?)  -- A = L L^T (or U^T U)
a:cholesky_inverse(upper?)  -- inverse from a Cholesky factor
b:cholesky_solve(factor, upper?)  -- solve A x = b from a Cholesky factor
a:cholesky_ex(opts?)   -- factor plus int32 info; opts.upper/check_errors
a:eigh(uplo?)          -- real symmetric eigenvalues (ascending) and vectors
a:eigvalsh(uplo?)      -- real symmetric eigenvalues only
a:svd(full_matrices?)  -- returns U, S, Vh
a:svdvals()            -- descending singular values, shape (*, min(m, n))
a:qr(mode?)            -- returns Q, R; mode defaults to "reduced"
a:householder_product(tau)  -- Q from compact Householder reflectors
a:lu()                 -- pivoted LU; returns P, L, U with A = P L U
a:lu_factor()          -- packed LU and 1-based pivots; raises if singular
a:lu_factor_ex(opts?)  -- packed LU, pivots, plus int32 info; opts.pivot/check_errors
a:lu_solve(pivots, b, opts?)  -- solve from a packed factorization
a:lu_unpack(pivots, opts?)    -- unpack packed LU into P, L, U
a:ldl_factor(opts?)    -- compact LDL^T factor and 1-based block pivots
a:ldl_factor_ex(opts?) -- factor, pivots, plus int32 info; opts.hermitian/check_errors
ld:ldl_solve(pivots, b, opts?)  -- solve from a compact LDL^T factor
a:pinv(opts?)          -- Moore-Penrose pseudoinverse; opts.atol/rtol
a:matrix_exp()         -- matrix exponential of square/batched matrices
a:matrix_power(n)      -- integer power; negative n inverts (floating only)
a:diagonal(offset?, dim1?, dim2?)  -- diagonal over two dimensions
a:tril(diagonal?)                 -- lower triangle; 0-based diagonal offset
a:triu(diagonal?)                 -- upper triangle; 0-based diagonal offset
a:trace()                         -- sum of the main diagonal of a 2-D matrix
a:diag(diagonal?)                 -- extract from a matrix or build from a vector
a:diag_embed(diagonal?)           -- embed the last dimension on a new diagonal
a:vander(N?, increasing?)          -- Vandermonde powers of the last dimension
a:cross(other, dim?)               -- 3-vector cross product along one dimension
a:vecdot(other, dim?)              -- batched vector dot along one dimension

torch.linalg.matmul(a, b)  -- broadcasting matrix product
torch.linalg.mm(a, b)      -- 2-D matrix product
torch.linalg.mv(a, v)      -- matrix-vector product
torch.linalg.outer(a, b)   -- outer product of two 1-D tensors
torch.kron(a, b)           -- Kronecker product
torch.cartesian_prod(t1, t2, ...)  -- combinations as rows
torch.linalg.inv(a)      torch.linalg.inv_ex(a, opts?)
torch.linalg.tensorinv(a, ind?)
torch.linalg.tensorsolve(a, b, dims?)
torch.linalg.det(a)      torch.linalg.slogdet(a)
torch.linalg.solve(a, b, opts?)  torch.linalg.cholesky(a) torch.linalg.svd(a)
torch.linalg.cholesky_inverse(a, upper?)  -- inverse from a Cholesky factor
torch.cholesky_solve(b, factor, upper?)   -- solve from a Cholesky factor
torch.linalg.solve_ex(a, b, opts?)  -- solution plus int32 info
torch.linalg.cholesky_ex(a, opts?)
torch.linalg.eigh(a, uplo?)  torch.linalg.eigvalsh(a, uplo?)
torch.linalg.svdvals(a)  -- descending singular values
torch.linalg.householder_product(a, tau)  -- Q from Householder reflectors
torch.linalg.solve_triangular(a, b, opts)
torch.linalg.qr(a)       torch.linalg.lstsq(a, b, opts?)  -- aliases
torch.linalg.lu(a)       torch.linalg.lu_factor(a)
torch.linalg.lu_factor_ex(a, opts?)
torch.linalg.lu_solve(lu, pivots, b, opts?)  torch.linalg.matrix_power(a, n)
torch.linalg.ldl_factor(a, opts?)  -- compact LDL^T factor
torch.linalg.ldl_factor_ex(a, opts?)  -- factor plus int32 info
torch.linalg.ldl_solve(ld, pivots, b, opts?)  -- solve from an LDL factor
torch.lu_unpack(lu, pivots, opts?)
torch.linalg.lu_unpack(lu, pivots, opts?)  -- alias of torch.lu_unpack
torch.diagonal(a, offset?, dim1?, dim2?)
torch.tril(a, diagonal?) / torch.triu(a, diagonal?)
torch.trace(a) / torch.diag(a, diagonal?)
torch.diag_embed(a, diagonal?) / torch.diagflat(a, diagonal?)
torch.tril_indices(row, col, diagonal?, opts?)
torch.triu_indices(row, col, diagonal?, opts?)
torch.tensordot(a, b, dims?)
torch.vander(x, N?, increasing?)          -- 1-D, decreasing by default
torch.cross(a, b, dim?)                   -- default dim is the last dimension
torch.linalg.diagonal(a, offset?, dim1?, dim2?)  -- last two dims by default
torch.linalg.vander(x, N?)                -- any rank, increasing by default
torch.linalg.cross(a, b, dim?)            -- last dimension by default
torch.linalg.vecdot(x, y, dim?)           -- last dimension by default
torch.linalg.multi_dot({a, b, c, ...})    -- optimal chain product
torch.linalg.pinv(a, opts?)
torch.linalg.matrix_exp(a)
```

`torch.linalg.matmul`, `torch.linalg.mm`, `torch.linalg.mv`, and
`torch.linalg.outer` mirror `torch.matmul`/`Tensor:matmul`,
`torch.mm`/`Tensor:mm`, `torch.mv`/`Tensor:mv`, and
`torch.outer`/`Tensor:outer`. They keep broadcasting, dtype promotion, and
autograd, and each rejects a non-tensor operand with a Lua error.

`torch.kron` / `Tensor:kron(other)` computes the Kronecker product with shapes
aligned from the right; missing leading dimensions count as 1 and each output
dimension is the product of the corresponding input dimensions. Scalars,
vectors, matrices, and higher-rank inputs are supported, and autograd flows
through the elementwise multiplication. `torch.cartesian_prod(t1, t2, ...)`
takes 1-D tensors and returns a `(total combinations, count)` tensor whose rows
enumerate the product with the first input varying slowest; an empty input
produces zero rows. It is differentiable through `repeat_interleave`, `tile`,
and `stack`.

`tensorinv` inverts a tensor with respect to `torch.tensordot`. Its first
`ind` dimensions (default 2) are grouped into `m` rows and the remaining
dimensions into `n` columns; the products must be equal, so the reshaped
`m x n` matrix is square and invertible. The result has the trailing
dimensions first and the grouped dimensions last:
`X:sizes() == A:sizes()[ind + 1:] + A:sizes()[:ind]`. In particular, a 2-D
matrix with `ind = 1` is an ordinary inverse. Only real float32/float64 inputs
are accepted, and autograd flows through the underlying inverse:

```lua
local a = torch.eye(12, {dtype = torch.float64}):reshape(3, 4, 2, 2, 3)
local inv = torch.linalg.tensorinv(a, 2)
print(inv:sizes())                    -- {2, 2, 3, 3, 4}
```

`tensorsolve` solves `torch.tensordot(A, X, X:dim()) = B`, returning `X`
with shape `A:sizes()[B:dim() + 1:]`. The first `B:dim()` dimensions of `A`
form the `m` equations and the remaining dimensions form the `n` unknowns;
the products must match. `dims` optionally moves the listed dimensions of `A`
to the end before grouping, which lets the contracted dimensions appear
anywhere in `A`. `B` may have any shape as long as it has `m` elements. Only
real float32/float64 inputs are accepted and both operands must share the
dtype; the solution is differentiable with respect to `A` and `B`:

```lua
local a = torch.eye(16, {dtype = torch.float64}):reshape(2, 2, 2, 2)
local b = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
local x = torch.linalg.tensorsolve(a, b)
print(x:sizes())                      -- {2, 2}
```

`householder_product` reconstructs the first `n` columns of a product of
Householder matrices from the compact representation used by LAPACK's `geqrf`:
a matrix `A` of shape `(*, m, n)` with `m >= n` and a `tau` tensor of shape
`(*, k)` with `k <= n`. The batch dimensions of `tau` must equal those of `A`,
the diagonal of `A` is ignored, and each reflector uses an implied leading
one. The result keeps the shape `(*, m, n)`. Only real float32/float64 inputs
are accepted and both operands must share the dtype. The operation is backed
by `at::linalg_householder_product`, so it is differentiable:

```lua
local a = torch.tensor({{9}, {3}, {4}}, {dtype = torch.float64})
local tau = torch.tensor({2 / 26}, {dtype = torch.float64})
print(torch.linalg.householder_product(a, tau):totable())
-- {{0.9230769230769231}, {-0.2307692307692308}, {-0.3076923076923077}}
```

`inv_ex` is the error-returning form of `inverse`. It returns the inverse
and an int32 `info` tensor whose entries are zero on success or positive when
the corresponding batch element is singular. Inputs may be batched square
matrices, `info` has the batch shape, and the inverse is still returned when
`info` is positive, matching PyTorch. Pass `{ check_errors = true }` to raise
on a singular matrix instead. Only float32/float64 are accepted; the inverse
is differentiable and the integer `info` stays detached:

```lua
local a = torch.tensor({{4, 2}, {2, 3}}, {dtype = torch.float64})
local inverse, info = torch.linalg.inv_ex(a)
print(info:item())                    -- 0
print(a:matmul(inverse):allclose(torch.eye(2, {dtype = torch.float64})))
```

`cholesky_ex` is the error-returning form of `cholesky`. It returns the
factor and an int32 `info` tensor whose entries are zero on success or the
order of the first leading minor that is not positive-definite. Inputs may be
batched square matrices; `info` then has the batch shape. By default the
factor is returned even when `info` is positive, matching PyTorch. Pass
`{ upper = true }` for `A = U^T U` or `{ check_errors = true }` to raise on a
failed factorization instead. Only float32/float64 are accepted, the factor is
differentiable, and the integer `info` stays detached:

```lua
local a = torch.tensor({{4, 2}, {2, 3}}, {dtype = torch.float64})
local l, info = torch.linalg.cholesky_ex(a)
print(info:item())                    -- 0
print(l:matmul(l:t()):allclose(a))    -- true
```

`cholesky_inverse` recovers the inverse of a real symmetric
positive-definite matrix from its Cholesky factor. Pass the lower factor `L`
from `A = L L^T` by default, or the upper factor `U` with `upper = true` for
`A = U^T U`. Square and batched float32/float64 factors are supported, the
result has the same shape as the factor, and the inverse is differentiable:

```lua
local a = torch.tensor({{4, 2}, {2, 3}}, {dtype = torch.float64})
local inverse = torch.linalg.cholesky_inverse(a:cholesky())
print(inverse:allclose(a:inverse()))  -- true
```

`torch.cholesky_inverse(a, upper?)` is the top-level alias. Only float32 and
float64 are accepted; sub-2-D, non-square, half, integer, and bool inputs are
rejected by the Lua layer.

`cholesky_solve` solves `A X = B` from a Cholesky factor of `A`. The method is
called on the right-hand side `B`: `B:cholesky_solve(L)` uses the lower factor
of `A = L L^T`, while `B:cholesky_solve(U, true)` uses the upper factor of
`A = U^T U`. `B` has shape `(*, n, k)`, so a single right-hand side is a
column matrix, and its batch dimensions may broadcast over the factor. Both
operands must share float32/float64 dtype, and the solution is differentiable:

```lua
local a = torch.tensor({{4, 2}, {2, 3}}, {dtype = torch.float64})
local b = torch.tensor({{1}, {2}}, {dtype = torch.float64})
local x = torch.cholesky_solve(b, a:cholesky())
print(x:allclose(a:solve(b)))         -- true
```

`torch.cholesky_solve(b, factor, upper?)` is the top-level alias. CPU LAPACK
has no half-precision kernel, so float16, integer, and bool right-hand sides
or factors are rejected.

`eigh` computes the eigenvalues and eigenvectors of real symmetric matrices
or batches of symmetric matrices. It reads the lower triangle by default;
pass `"U"` (or `"u"`) to read the upper triangle instead. The first result
holds the eigenvalues in ascending order with shape `(*, n)`, and the second
holds orthonormal eigenvectors as columns with shape `(*, n, n)`, so
`V @ diag(w) @ V^T == A`. `eigvalsh` returns only the ascending eigenvalues,
using a path that does not materialize the eigenvectors:

```lua
local a = torch.tensor({{2, 1}, {1, 3}}, {dtype = torch.float64})
local w, v = torch.linalg.eigh(a)
print(w:totable())                   -- {1.3819660112501, 3.6180339887499}
print(torch.linalg.eigvalsh(a):totable())  -- same values
```

Both forms accept batched and empty square matrices, require float32 or
float64, and support autograd through the eigenvalues and eigenvectors.
`UPLO` must be `"L"` or `"U"` (case-insensitive); other values, non-square
inputs, and sub-2-D tensors are rejected by the Lua layer.

`svdvals` returns only the singular values of a matrix or a batch of
matrices, in descending order, with shape `(*, min(m, n))`. It is equivalent
to the `S` returned by `svd(false)` but avoids materializing `U` and `Vh`.
CPU LAPACK only provides the real float32/float64 path, so float16, integer,
and bool inputs are rejected; singular values are differentiable with respect
to the input:

```lua
local a = torch.tensor({{1, 2}, {3, 4}, {5, 6}}, {dtype = torch.float64})
print(torch.linalg.svdvals(a):sizes())  -- {2}
```

`slogdet` computes the signed log-determinant without forming the determinant,
so it stays finite for very large or small magnitudes. `sign` is `-1`, `0`, or
`1`; `logabsdet` is real. For a square matrix `exp(logabsdet) * sign == det`,
and gradients flow through `logabsdet` as `A^{-T}`. Batched inputs return
`sign` and `logabsdet` with the batch dimensions; square matrices are required.

`solve` solves `A X = B` by default. Pass `{ left = false }` to solve
`X A = B`: the right-hand side must then have at least two dimensions with
shape `(*, k, n)`, and the result has the same shape. The right solve is
computed through a transposed left solve, so it stays differentiable with
respect to both the matrix and the right-hand side. Vector right-hand sides are
accepted only for left solves:

```lua
local a = torch.tensor({{2, 0}, {1, 3}}, {dtype = torch.float64})
local b = torch.tensor({{2, 4}}, {dtype = torch.float64})
print(a:solve(b, { left = false }):totable())   -- {{0.3333, 1.3333}}
print(torch.linalg.solve(a, b, { left = false }):allclose(
  a:solve(b, { left = false })))
```

`solve_ex` is the error-reporting form of `solve`. It returns the solution and
an int32 `info` tensor whose entries are zero on success or positive when the
corresponding batch element is singular. `left` defaults to true (`A X = B`)
and false solves `X A = B`; `check_errors` defaults to false, so a singular
system is solved and reported through `info` instead of raising. A 1-D
right-hand side is accepted for a left solve; otherwise `B` has shape
`(*, n, k)` for a left solve and `(*, k, n)` for a right solve. Batch
dimensions broadcast, both operands must share float32/float64 dtype, and the
solution is differentiable:

```lua
local a = torch.tensor({{4, 3}, {2, 1}}, {dtype = torch.float64})
local b = torch.tensor({{1}, {2}}, {dtype = torch.float64})
local x, info = torch.linalg.solve_ex(a, b)
print(info:item(), x:totable())       -- 0, {{2.5}, {-3}}
```

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

`lu_factor_ex` is the error-reporting form of `lu_factor`. It returns
`(LU, pivots, info)` with the same packed factor and int32 1-based pivots,
plus an int32 `info` tensor whose entries are zero on success or positive when
the corresponding batch element is singular. `pivot` defaults to true and
`check_errors` defaults to false, matching PyTorch: a singular factorization is
returned by default, and `{ check_errors = true }` raises instead. The CPU
LAPACK path has no unpivoted kernel, so `{ pivot = false }` is rejected
explicitly. Square, rectangular, batched, and empty float32/float64 inputs are
accepted, and the factor is differentiable:

```lua
local a = torch.tensor({{4, 3}, {2, 1}}, {dtype = torch.float64})
local lu, pivots, info = torch.linalg.lu_factor_ex(a)
print(info:item())                    -- 0
print(lu:lu_solve(pivots, torch.tensor({{1}, {2}},
  {dtype = torch.float64})):totable())  -- {{2.5}, {-3}}
```

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

`lu_unpack` turns the compact `(LU, pivots)` pair back into `P`, `L`, and
`U` with `A = P L U`. It accepts the same square, rectangular, batched, and
empty factors as `lu_factor`; `P` has shape `(*, m, m)`, `L` has shape
`(*, m, k)`, and `U` has shape `(*, k, n)` for an input of shape `(*, m, n)`.
The integer pivots are validated as 1-based row indices, and `P` stays
detached. `L` and `U` are built with differentiable tensor arithmetic, so
gradients flow through the packed factor. `opts.unpack_data = false` returns
empty `L`/`U`, and `opts.unpack_pivots = false` returns an empty `P`.
`torch.linalg.lu_unpack` is an alias of `torch.lu_unpack`:

```lua
local a = torch.tensor({{4, 3}, {2, 1}}, {dtype = torch.float64})
local lu, pivots = a:lu_factor()
local p, l, u = torch.lu_unpack(lu, pivots)
print(p:matmul(l):matmul(u):totable())   -- {{4, 3}, {2, 1}}
```

`ldl_factor` computes a compact LDL^T factorization of a real symmetric matrix
or a batch of symmetric matrices. It returns `LD`, which has the same shape as
`A`, and int32 1-based `pivots` with shape `(*, n)`. Following LAPACK's
`sytrf`, the strict lower triangle of `LD` stores the unit lower-triangular
multipliers and the diagonal stores the 1x1 and 2x2 diagonal blocks of `D`.
For a factorization with only 1x1 pivots, `LD` encodes `A = L D L^T`. A
singular matrix raises; use `ldl_factor_ex` to report failure through `info`
instead. The CPU path accepts float32/float64 only, supports batches and empty
matrices, and has no implemented derivative for the factorization:

```lua
local a = torch.tensor({{4, 2}, {2, 3}}, {dtype = torch.float64})
local ld, pivots = torch.linalg.ldl_factor(a)
print(ld:totable())     -- {{4, 0}, {0.5, 2}}
print(pivots:totable()) -- {1, 2}
```

`ldl_factor_ex` is the error-reporting form of `ldl_factor`. It returns
`(LD, pivots, info)` where the int32 `info` tensor is zero on success and
positive when the corresponding batch element is singular. `hermitian`
defaults to false and `check_errors` defaults to false, so a singular matrix
is returned with a positive `info` instead of raising; pass
`{ check_errors = true }` to raise. The `LD` and `pivots` outputs match
`ldl_factor` for nonsingular inputs, and batches and empty matrices are
supported:

```lua
local a = torch.tensor({{4, 2}, {2, 3}}, {dtype = torch.float64})
local ld, pivots, info = torch.linalg.ldl_factor_ex(a)
print(info:item())                    -- 0
```

`ldl_solve` solves `A X = B` from the `(LD, pivots)` pair produced by
`ldl_factor` or `ldl_factor_ex`. `B` has shape `(*, n, k)`, so a single
right-hand side is a column matrix, and its batch dimensions may broadcast
over the factor. The pivots must be int32 and match the factor's batch shape
and length, `LD` must be a square float32/float64 matrix or batch, and both
operands must share the dtype. For real inputs `hermitian` has no effect. The
CPU path does not implement a derivative for this solve:

```lua
local a = torch.tensor({{4, 2}, {2, 3}}, {dtype = torch.float64})
local ld, pivots = torch.linalg.ldl_factor(a)
local b = torch.tensor({{1}, {2}}, {dtype = torch.float64})
print(torch.linalg.ldl_solve(ld, pivots, b):totable())  -- {{-0.125}, {0.75}}
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

`matrix_exp` computes the matrix exponential through ATen's scaling-and-squaring
implementation. Unlike an elementwise `exp`, it solves the matrix ODE and
satisfies `exp(A) exp(-A) = I` and `det(exp(A)) = exp(tr A)`. An input of
shape `(*, n, n)` returns the same shape, including empty matrices, and the
result is differentiable with respect to the input:

```lua
local a = torch.tensor({{0, 1}, {0, 0}}, {dtype = torch.float64})
print(a:matrix_exp():totable())          -- {{1, 1}, {0, 1}}
```

Only real float32/float64 inputs are accepted. float16, integer, and bool
inputs are rejected with Lua errors: ATen's CPU half kernel is not numerically
reliable for this operation (small inputs can return NaN or large negative
values), so luatorch does not expose it.

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

`diagonal` extracts a diagonal from any two dimensions. The selected diagonal
is appended as the last output dimension, while all other dimensions keep
their order; `offset` is positive above the main diagonal and negative below
it. `torch.diagonal` and `Tensor:diagonal` default to dimensions 1 and 2,
whereas `torch.linalg.diagonal` defaults to the last two dimensions. Integer
and floating dtypes are accepted, but bool is rejected.
The result is differentiable with respect to the selected entries:

```lua
local a = torch.tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
  {dtype = torch.float64})
print(a:diagonal(1):totable())      -- {2, 6}
print(a:diagonal(-1):totable())     -- {4, 8}
```

`tril` and `triu` return the lower or upper triangular part of a matrix or
batch of matrices, zeroing the opposite half. `diagonal` defaults to `0` (the
main diagonal), is positive above it and negative below it. The mask broadcasts
over batch dimensions, every dtype is accepted (integer and bool entries are
zeroed with the dtype's zero), and the result is differentiable with zero
gradient in the zeroed half. Inputs must have at least two dimensions:

```lua
local a = torch.tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
  {dtype = torch.float64})
print(a:tril())       -- [[1, 0, 0], [4, 5, 0], [7, 8, 9]]
print(a:triu(1))      -- [[0, 2, 3], [0, 0, 6], [0, 0, 0]]
print(torch.tril(a):allclose(a:tril()))
```

`trace` sums the main diagonal of a 2-D matrix and returns a 0-dim tensor. It
is differentiable and accepts integer and floating dtypes:

```lua
print(torch.tensor({{1, 2}, {3, 4}}):trace():item())   -- 5
```

`diag` mirrors `torch.diag`. For a 2-D input it extracts a diagonal and returns
a 1-D tensor; for a 1-D input it builds a square matrix with the vector on the
requested diagonal, padding with zeros. The `diagonal` argument uses the same
0-based convention as `tril`/`triu`, and both directions are differentiable.
One-dimensional construction accepts every dtype, including bool; two-
dimensional extraction follows `diagonal` and accepts integer and floating
dtypes:

```lua
local v = torch.tensor({1, 2, 3}, {dtype = torch.float64})
print(v:diag(1))      -- a 4x4 matrix with 1, 2, 3 above the main diagonal
print(v:diag():trace():item())  -- 6
```

`diag_embed` is the batched construction form: for an input of shape `(*, n)`
it returns `(*, size, size)` with `size = n + |offset|`, placing each vector on
the requested diagonal and padding with zeros. The offset is 0-based and the
leading dimensions are treated as batch dimensions; 1-D, 2-D, empty, integer,
and bool inputs are supported, and the result is differentiable. Only the last
two output dimensions are used:

```lua
local batch = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
print(batch:diag_embed())    -- two 2x2 diagonal matrices
print(torch.diag_embed(batch, -1):sizes())  -- {2, 3, 3}
```

`torch.diagflat(a, diagonal?)` flattens its input and builds a 2-D diagonal
matrix from the resulting vector, so `torch.diagflat(batch)` produces one
4x4 matrix for a 2x2 input.

`tril_indices(row, col, diagonal?, opts?)` and
`triu_indices(row, col, diagonal?, opts?)` return the coordinates of the
selected triangle as a `2 x N` integer tensor in row-major order. Coordinates
are **1-based**, matching the rest of the Lua surface (PyTorch returns
0-based coordinates). The offset defaults to 0, and `opts.dtype` selects the
output dtype (default `int64`; `opts.device` must be `"cpu"`):

```lua
print(torch.tril_indices(3, 3):totable())
-- {{1, 2, 2, 3, 3, 3}, {1, 1, 2, 1, 2, 3}}
print(torch.triu_indices(3, 3, 1):totable())
-- {{1, 1, 2}, {2, 3, 3}}
```

`vander` builds a Vandermonde matrix. `torch.linalg.vander(x, N?)` accepts any
rank, uses `N = x:size(-1)` by default, and appends the powers
`x^0 .. x^(N-1)` (increasing order). `torch.vander(x, N?, increasing?)`
requires a 1-D tensor and defaults to the historical decreasing order. Both
forms preserve autograd through the powers. `N` must be a non-negative
integer; `N = 0` returns an empty last dimension.

`cross` computes the cross product of two real 3-vectors along `dim`, which
defaults to the last dimension. The inputs must have the same rank, the same
dtype, and length 3 along `dim`; their other dimensions broadcast. Integer and
floating dtypes are supported, bool is rejected, and autograd flows to both
inputs. `torch.cross` and `torch.linalg.cross` are aliases.

`vecdot` computes the dot product of two batches of vectors along `dim`.
The reduction dimension is resolved against the broadcast rank, so the inputs
follow ordinary NumPy broadcasting before the elementwise product is summed;
the result has the broadcast shape with `dim` removed. Inputs must share one
float16/float32/float64 dtype; integer and bool inputs are rejected. The
operation is built from `mul` and `sum`, so autograd flows to both inputs:

```lua
local x = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
local y = torch.tensor({{5, 6}, {7, 8}}, {dtype = torch.float64})
print(torch.linalg.vecdot(x, y):totable())   -- {17, 53}
print(x:vecdot(y, 1):totable())              -- {23, 34}
```

`torch.linalg.matrix_norm(a, ord?, dim1?, dim2?, keepdim?)` supports the
matrix orders `"fro"`, `"nuc"`, `1`, `-1`, `2`, `-2`, `math.huge`, and
`-math.huge`, with `"fro"` as the default. Unlike the older 2-D-only form it
accepts batched inputs, reduces any dimension pair (`dim1`/`dim2` default to
the last two dimensions), and keeps the reduced dimensions as size one when
`keepdim` is true. The `2`, `-2`, and `"nuc"` orders use the SVD path and
therefore require float32/float64; the other orders use reductions and accept
float16 as well. For a plain 2-D input with the default dimensions and no
`keepdim`, the historical Lua-number result is preserved:

```lua
local a = torch.tensor({{{1, 2}, {3, 4}}, {{2, 0}, {0, 1}}},
  {dtype = torch.float64})
print(torch.linalg.matrix_norm(a, "fro"):totable())  -- {5.477, 2.236}
print(torch.linalg.matrix_norm(a, "nuc"):totable())  -- {5.831, 3}
local b = torch.arange(1, 25, 1, {dtype = torch.float64}):reshape(2, 3, 4)
print(torch.linalg.matrix_norm(b, 1, 1, 3, true):sizes())  -- {1, 3, 1}
```

`torch.linalg.norm(a, ord?, dim?, keepdim?)` is the umbrella dispatcher that
matches PyTorch's deprecated `linalg.norm` behavior. With no `dim` and no
`ord` it computes the flattened 2-norm (Frobenius for matrices). With no
`dim` but an explicit `ord`, the input must be 1-D or 2-D: 1-D uses a vector
norm and 2-D uses a matrix norm. An integer `dim` always selects a vector
norm, while a one- or two-element table selects a vector or matrix norm over
those dimensions. The vector orders `0`, positive and negative finite
numbers, `math.huge`, and `-math.huge` are supported; matrix orders are those
of `matrix_norm`. The result is a tensor, and `keepdim` keeps every reduced
dimension as size one (including a fully flattened norm):

```lua
local a = torch.arange(1, 7, 1, {dtype = torch.float64}):reshape(2, 3)
print(torch.linalg.norm(a):item())                 -- 9.5393920141695
print(torch.linalg.norm(a, 2):item())              -- spectral norm
print(torch.linalg.norm(a, 1, {1, 2}):item())      -- matrix 1-norm, 9
print(torch.linalg.norm(a, nil, 1):totable())      -- column 2-norms
print(torch.linalg.norm(a, "fro", {1, 2}, true):sizes())  -- {1, 1}
```

`torch.linalg.cond(a, p?)` computes the condition number of a matrix or batch
of matrices. The spectral orders `2` and `-2` use the largest and smallest
singular values and accept rectangular inputs; every other order multiplies a
matrix norm by the same norm of the inverse, so square matrices are required.
Supported orders are `2` (the default), `-2` (smallest/largest singular
value), `1`, `-1`, `math.huge`, `-math.huge`, `"fro"`, and `"nuc"`
(case-insensitive). A plain 2-D input keeps the historical Lua-number result,
while batches return a tensor with the batch shape. Only float32 and float64
are accepted:

```lua
local a = torch.tensor({{{2, 0}, {0, 1}}, {{1, 0}, {0, 1}}},
  {dtype = torch.float64})
print(torch.linalg.cond(a, 2):totable())    -- {2, 1}
print(torch.linalg.cond(a, -2):totable())   -- {0.5, 1}
print(torch.linalg.cond(a, "nuc"):totable()) -- {4.5, 4}
```

`torch.linalg.matrix_rank(a, tol?, hermitian?)` returns the numerical rank of
a matrix or batch of matrices. The default tolerance is
`max(m, n) * eps(dtype) * largest singular value`; a numeric `tol` is used as
an absolute cutoff, so singular values strictly greater than `tol` count.
Alternatively pass an options table:
`torch.linalg.matrix_rank(a, { tol = t, rtol = r, hermitian = h })`. `rtol`
uses the relative cutoff `rtol * largest`; supplying both `tol` and `rtol` is
an error. `hermitian = true` requires square input and uses the absolute
eigenvalues of the real symmetric matrix instead of its singular values. A
plain 2-D input still returns the historical Lua number, while batched inputs
return an `int64` tensor with the batch shape. Only float32 and float64 are
accepted, matching the SVD and symmetric-eigenvalue CPU paths:

```lua
local a = torch.tensor({{{1, 0}, {0, 1}}, {{1, 1}, {1, 1}}},
  {dtype = torch.float64})
print(torch.linalg.matrix_rank(a):totable())          -- {2, 1}
print(torch.linalg.matrix_rank(a, 1e-12, true):totable())  -- {2, 1}
print(torch.linalg.matrix_rank(a, { rtol = 1e-6 }):totable())  -- {2, 1}
```

`torch.linalg.multi_dot({a, b, ...})` multiplies two or more matrices with a
dynamic-programming parenthesization that minimizes scalar multiplications.
The first and last operands may be 1-D vectors (treated as a row and column
respectively); all middle operands must be 2-D, and batched inputs are not
supported. All operands must share a float16/float32/float64 dtype. The result
is the mathematically identical chain product, and autograd flows through every
`matmul`:

```lua
local a = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
local b = torch.tensor({{5, 6}, {7, 8}}, {dtype = torch.float64})
print(torch.linalg.multi_dot({a, b}):totable())  -- {{19, 22}, {43, 50}}
```

`torch.tensordot(a, b, dims?)` contracts tensor dimensions. An integer `dims`
(default 2) contracts the last `dims` dimensions of `a` with the first `dims`
of `b`; a `{a_dims, b_dims}` pair selects arbitrary (including negative)
dimensions. The paired dimensions must have equal sizes, both operands must
share one dtype, and the free dimensions appear in the result in input order:
`a`'s free dimensions followed by `b`'s. The contraction is a
permute/reshape/matmul, so autograd flows to both operands:

```lua
local a = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
local b = torch.tensor({{5, 6}, {7, 8}}, {dtype = torch.float64})
print(torch.tensordot(a, b, 1):totable())   -- {{19, 22}, {43, 50}}
print(torch.tensordot(a, b, 0):sizes())     -- {2, 2, 2, 2} (outer product)
```

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
t:roll(shifts, dims?)           -- circular shift; no dims flattens first
t:tile(reps)                    -- repeat each dimension by reps
t:repeat_interleave(repeats, dim?) -- repeat elements or slices
t:split(size_or_sections, dim?)   -- fixed-size or explicit sections
t:chunk(chunks, dim?)             -- at most `chunks` roughly equal parts
t:unbind(dim?)                    -- slices with one dimension removed
t:swapaxes(dim1, dim2)            -- transpose two dimensions
t:moveaxis(source, destination)   -- move one or more dimensions
torch.atleast_1d(...) / torch.atleast_2d(...) / torch.atleast_3d(...)
torch.hstack({...}) / torch.vstack({...}) / torch.dstack({...})
torch.column_stack({...}) / torch.row_stack({...})
torch.cat({t1, t2}, dim?)      -- default dim = 1
torch.stack({t1, t2}, dim?)    -- default dim = 1
torch.broadcast_shapes({2, 1}, {3})  -- {2, 3}
torch.broadcast_tensors(t1, t2)      -- expanded views, multiple returns
```

`expand` follows PyTorch broadcasting rules (including `-1` to keep a size).
`broadcast_shapes` aligns trailing dimensions and accepts varargs of shape
tables, a single table of shape tables, or no arguments (empty shape).
`broadcast_tensors` computes that common shape and returns one expanded view per
input; it also accepts a single table of tensors.
`roll` accepts an integer shift or equal-length shift/dim lists; without
`dims` it flattens, rolls, and reshapes. `tile` pads missing repeats with 1 and
promotes the input with leading size-1 dimensions when `reps` is longer.
`repeat_interleave` takes a non-negative scalar or a 1-D integer tensor of
repeats; without `dim` it flattens the input first. All three are
differentiable through their underlying views and gathers.

`split`, `chunk`, and `unbind` return Lua arrays of views. `split` takes a
positive chunk size or a list of section sizes that must sum to the dimension;
`chunk` computes `ceil(size/chunks)` and may return fewer than `chunks` parts;
`unbind` removes the chosen dimension. `swapaxes` is a transpose alias and
`moveaxis` accepts integer or equal-length source/destination lists (also
available as `torch.movedim`). `atleast_1d`, `atleast_2d`, and `atleast_3d`
return each input at the requested minimum rank, leaving already-large inputs
unchanged; they return multiple values.

`hstack` concatenates 1-D inputs along their only axis and higher-rank inputs
along the second dimension. `vstack` promotes each input to at least 2-D and
concatenates rows; `dstack` promotes to at least 3-D and concatenates along the
third dimension; `column_stack` stacks 1-D inputs as columns and otherwise
concatenates along the second dimension; `row_stack` is an alias of `vstack`.

## Reductions

```lua
t:sum(dim?, keepdim?)
t:mean(dim?, keepdim?)
t:prod(dim?, keepdim?)
t:max(dim?, keepdim?)          -- values[, indices] when dim is given
t:min(dim?, keepdim?)          -- values[, indices] when dim is given
t:aminmax(dim?, keepdim?)      -- minimum and maximum values only
t:amin(dim?, keepdim?)         -- min values; dim may be a table
t:amax(dim?, keepdim?)         -- max values; dim may be a table
t:count_nonzero(dim?)          -- number of non-zero entries (int64)
t:nansum(dim?, keepdim?)       -- sum ignoring NaN
t:nanmean(dim?, keepdim?)      -- mean ignoring NaN (floating dtypes)
t:nanvar(dim?, unbiased?, keepdim?)  -- variance ignoring NaN
t:nanstd(dim?, unbiased?, keepdim?)  -- standard deviation ignoring NaN
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

`aminmax` returns the minimum and maximum values as two tensors without
materializing the indices. With no dimension both results are 0-dim and
`keepdim` is ignored; with an integer dimension (negative counts from the end)
both have the reduced shape, or the kept shape when `keepdim` is true.
`torch.aminmax(t, dim?, keepdim?)` is the module-level form. NaN propagates
through both values, like `min`/`max`.

`amin` and `amax` return only the minimum or maximum values. `dim` may be an
integer, a negative dimension, or a non-empty table of distinct dimensions;
tables are reduced from the highest index down so the remaining dimensions
stay valid when `keepdim` is false. `torch.amin` and `torch.amax` are the
module-level forms:

```lua
local t = torch.tensor({{1, 5, 3}, {4, 2, 6}}, {dtype = torch.float64})
print(t:amin(1):totable())   -- {1, 2, 3}
print(t:amax(2):totable())   -- {5, 6}
print(t:amin({1, 2}):item()) -- 1
```

`count_nonzero` returns an int64 tensor with the number of entries that are
not zero (NaN counts as non-zero, matching PyTorch). With no dimension the
result is 0-dim; `dim` may be an integer or a non-empty table of distinct
dimensions, in which case the reduced dimensions are removed:

```lua
local t = torch.tensor({{1, 0, 3}, {0, 5, 0}})
t:count_nonzero():item()        -- 3
t:count_nonzero(1):totable()    -- {1, 1, 1}
t:count_nonzero({1, 2}):item()  -- 3
```

`nansum` and `nanmean` reduce while treating NaNs as missing. `nansum` returns
zero for an all-NaN slice and accepts every dtype; `nanmean` divides by the
number of non-NaN entries, returns NaN for an all-NaN slice, and requires a
floating-point dtype. Infinities are ordinary values, and gradients pass only
through non-NaN entries:

```lua
local t = torch.tensor({{0 / 0, 2}, {3, 4}}, {dtype = torch.float64})
print(t:nansum():item())       -- 9
print(t:nanmean():item())      -- 3
print(t:nanmean(2):totable())  -- {2, 3.5}
```

`nanvar` and `nanstd` extend the same missing-data rule to variance and its
square root. The valid count excludes NaNs, `unbiased` defaults to true, and a
slice with too few valid entries (none, or one for the unbiased denominator)
returns NaN. Floating-point dtypes are required, and gradients pass only
through non-NaN entries:

```lua
local t = torch.tensor({1, 0 / 0, 3}, {dtype = torch.float64})
print(t:nanvar():item())                 -- 2
print(t:nanvar(nil, false):item())       -- 1
print(t:nanstd():item())                 -- 1.4142
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
a:isclose(b, rtol?, atol?, equal_nan?)   -- elementwise bool tensor
a:allclose(b, rtol?, atol?, equal_nan?)  -- one bool for the whole tensor
torch.equal(a, b)
torch.isclose(a, b, rtol?, atol?, equal_nan?)
torch.allclose(a, b, rtol?, atol?, equal_nan?)
```

`isclose` compares elementwise with `rtol = 1e-5` and `atol = 1e-8` by
default: two finite values are close when `|a - b| <= atol + rtol * |b|`.
Exact equality is also close, so matching infinities compare close while
`+inf` and `-inf` do not. Matching NaNs are close only when `equal_nan = true`.
The result has the broadcast shape and is a detached bool tensor; integer
tensors are accepted, bool tensors are rejected. `allclose` is
`isclose(...):all():item()` and keeps its historical default tolerances.

```lua
local a = torch.tensor({1, 2, 3})
local b = torch.tensor({1, 2.0000001, 4})
print(a:isclose(b))                     -- tensor([true, true, false])
print(a:allclose(b))                    -- false
print(torch.tensor({0/0}):allclose(torch.tensor({0/0}), nil, nil, true))
```

## NaN and Inf predicates

```lua
torch.isnan(t)          -- bool tensor, true where the value is NaN
torch.isinf(t)          -- bool tensor, true where the value is +/-Inf
torch.isposinf(t)       -- bool tensor, true only for +Inf
torch.isneginf(t)       -- bool tensor, true only for -Inf
torch.isfinite(t)       -- bool tensor, true where neither NaN nor +/-Inf
                        -- integer and bool inputs are always finite
t:isnan()               -- method aliases
t:isinf()
t:isposinf()
t:isneginf()
t:isfinite()
```

The result has the same shape as the input and never participates in autograd:
predicates return a detached bool tensor even when the input requires grad.
Float16/32/64 values are checked; integer and bool tensors produce all-true
masks, like PyTorch. `isposinf` and `isneginf` split the infinity mask by sign:
`+Inf` is positive and `-Inf` negative, while NaN and finite values match
neither. `torch.isposinf`/`torch.isneginf` are the module-level forms.

`nan_to_num` replaces `NaN`, `+Inf`, and `-Inf` with finite numbers:

```lua
t:nan_to_num(nan?, posinf?, neginf?)
torch.nan_to_num(t, nan?, posinf?, neginf?)
```

For float16/32/64 the defaults are `nan = 0`, `posinf =` the largest finite
value of the dtype, and `neginf =` the smallest finite value. Supplying a
number overrides one replacement; `nil` keeps the default. The input is not
modified, the dtype is preserved, and integer/bool tensors are returned as a
clone because they cannot hold non-finite values. The operation is
differentiable: gradients pass through finite entries and are zero at replaced
entries, matching PyTorch.

```lua
local t = torch.tensor({1, 0/0, math.huge}, {dtype = torch.float64})
print(t:nan_to_num())                       -- tensor([1., 0., 1.7977e+308])
print(t:nan_to_num(-1, 7, -7))              -- tensor([1., -1., 7.])
```

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
`clamp`, `clamp_min`, `clamp_max`, `logit`, `relu`, `sigmoid`, `tanh`,
`softplus`, `elu`, `selu`, `silu`, `gelu`, `exp`, `expm1`, `log`, `log2`,
`log10`, `erf`, `erfc`, `sinc`, `sqrt`, `abs` are thin wrappers around the
corresponding tensor methods.

Other module helpers:

```lua
torch.is_tensor(x)
torch.manual_seed(seed)
torch.set_default_dtype(dtype)
torch.get_default_dtype()
torch.dtype_name(dtype)
torch.get_num_threads() / torch.set_num_threads(n)
torch.sort / topk / unique / median / quantile / bincount / histc / nonzero / gather / scatter / multinomial
torch.argsort(t, dim?, descending?) / torch.diff(t, n?, dim?)
torch.roll(t, shifts, dims?) / torch.tile(t, reps)
torch.repeat_interleave(t, repeats, dim?)
torch.split(t, size_or_sections, dim?) / torch.chunk(t, chunks, dim?)
torch.unbind(t, dim?)
torch.swapaxes(t, d1, d2) / torch.moveaxis(t, src, dst) / torch.movedim(...)
torch.atleast_1d(...) / torch.atleast_2d(...) / torch.atleast_3d(...)
torch.hstack({...}) / torch.vstack({...}) / torch.dstack({...})
torch.column_stack({...}) / torch.row_stack({...})
torch.take(t, index)             -- flattened 1-based gather
torch.searchsorted(sorted, values, right?)  -- 1-based insertion positions
torch.bucketize(input, boundaries, right?)   -- 1-based bucket indices
torch.aminmax(t, dim?, keepdim?) -- min and max values in one call
torch.amin(t, dim?, keepdim?) / torch.amax(t, dim?, keepdim?)
torch.count_nonzero(t, dim?)     -- non-zero count (int64; dim int or table)
torch.nansum(t, dim?, keepdim?) / torch.nanmean(t, dim?, keepdim?)
torch.nanvar(t, dim?, unbiased?, keepdim?) / torch.nanstd(...)
torch.addcmul(input, t1, t2, value?) / torch.addcdiv(input, t1, t2, value?)
torch.lerp(input, end_tensor, weight)
torch.logaddexp(a, b) / torch.logaddexp2(a, b) / torch.hypot(a, b)
torch.frac(t)
torch.logit(t, eps?) / torch.clamp_min(t, v) / torch.clamp_max(t, v)
torch.softplus(t, beta?, threshold?) / torch.elu(t, alpha?)
torch.selu(t) / torch.silu(t) / torch.gelu(t, approximate?)
torch.ger(a, b)                  -- outer product of two vectors
torch.kron(a, b) / torch.cartesian_prod(t1, t2, ...)
torch.tril(a, diagonal?) / torch.triu(a, diagonal?)
torch.trace(a) / torch.diag(a, diagonal?)
torch.diag_embed(a, diagonal?) / torch.diagflat(a, diagonal?)
torch.tril_indices(row, col, diagonal?, opts?)
torch.triu_indices(row, col, diagonal?, opts?)
torch.tensordot(a, b, dims?)
torch.broadcast_shapes(...) / torch.broadcast_tensors(...)
torch.einsum(equation, ...)      -- validated Einstein summation
torch.trapezoid(y, x?, opts?)    -- trapezoidal integration along one dim
torch.isnan(t) / torch.isinf(t) / torch.isfinite(t)  -- bool predicates
torch.isposinf(t) / torch.isneginf(t)  -- signed-infinity predicates
torch.nan_to_num(t, nan?, posinf?, neginf?)  -- replace non-finite values
torch.isclose(a, b, rtol?, atol?, equal_nan?)  -- elementwise bool tensor
torch.linalg.matmul(a, b) / mm(a, b) / mv(a, b) / outer(a, b)
torch.linalg.matrix_norm(a, ord?, dim1?, dim2?, keepdim?)
                                  -- fro, nuc, 2, -2, 1, -1, inf, -inf
torch.linalg.norm(a, ord?, dim?, keepdim?)  -- vector/matrix norm dispatcher
torch.linalg.solve(a, b, opts?)  -- solve A x = b; opts.left=false for x A = b
torch.linalg.solve_ex(a, b, opts?)  -- solution plus int32 info
torch.linalg.lu_factor_ex(a, opts?)  -- packed LU, pivots, plus int32 info
torch.linalg.ldl_factor(a, opts?) / torch.linalg.ldl_factor_ex(a, opts?)
torch.linalg.ldl_solve(ld, pivots, b, opts?)  -- solve from an LDL factor
torch.linalg.cholesky_ex(a, opts?)  -- factor plus int32 info
torch.linalg.cholesky_inverse(a, upper?)  -- inverse from a Cholesky factor
torch.cholesky_solve(b, factor, upper?)   -- solve from a Cholesky factor
torch.linalg.eigh(a, uplo?) / torch.linalg.eigvalsh(a, uplo?)
torch.linalg.inv_ex(a, opts?)    -- inverse plus int32 info
torch.linalg.cond(a, p?)         -- 2, -2, 1, -1, inf, -inf, fro, nuc
torch.linalg.matrix_rank(a, tol?, hermitian?)
torch.linalg.matrix_rank(a, { tol = t, rtol = r, hermitian = h })
                                  -- batched int64 ranks; 2-D returns a number
torch.linalg.matrix_power(a, n)  -- integer matrix power
torch.linalg.vector_norm(a, ord?, dim?, keepdim?)
torch.linalg.slogdet(a)          -- returns sign, logabsdet
torch.linalg.lstsq(a, b, opts?)  -- returns x, residuals, rank, singular values
torch.linalg.pinv(a, opts?)      -- Moore-Penrose pseudoinverse
torch.linalg.matrix_exp(a)       -- matrix exponential
torch.lu_unpack(lu, pivots, opts?)  -- P, L, U from a packed factorization
torch.linalg.lu_unpack(lu, pivots, opts?)  -- alias of torch.lu_unpack
torch.diagonal(a, offset?, dim1?, dim2?) / torch.linalg.diagonal(a, ...)
torch.vander(x, N?, increasing?) / torch.linalg.vander(x, N?)
torch.cross(a, b, dim?) / torch.linalg.cross(a, b, dim?)
torch.linalg.multi_dot({a, b, ...})
torch.mv / outer / addmm / baddbmm
torch.ALL_DIMS            -- -1000 sentinel (internal, exposed for clarity)
torch.profiler            -- opt-in tensor-method profiler (see docs/profiler.md)
torch.VERSION             -- luatorch version string ("0.34.0")
torch.version             -- libtorch version string
```
