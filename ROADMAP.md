# luatorch roadmap

The project's goal is a Torch7-class Lua tensor stack that is faster than eager
PyTorch where execution strategy matters: fused elementwise kernels, low
dispatch overhead, and explicit memory management, using ATen for the heavy
BLAS/conv kernels. This roadmap is intentionally
concrete: each milestone should leave `scripts/test.sh` green and documented.

## v0.1 — tensor foundation (done)

- [x] C ABI shim over `at::Tensor` with exception-safe error reporting
- [x] LuaJIT FFI loader and PyTorch-style `torch` module
- [x] Dtypes: float16/32/64, int8/16/32/64, uint8, bool
- [x] Creation, broadcasting math, ~25 unary ops, matmul family
- [x] Shape ops, slicing, Boolean masking, `index_select`, `cat`/`stack`
- [x] Reductions with dim/keepdim, 1-based indices
- [x] Autograd: `requires_grad`, `backward`, `grad`, `detach`, `no_grad`
- [x] PyTorch `.pt` serialization
- [x] luafun integration (`torch.fun`, `Tensor:iter/map/filter/reduce`, `t:fun()`)
- [x] 261-test suite, benchmarks, API/design docs
- [x] Moses-vs-luafun benchmark documenting the functional-layer choice

## v0.2 — neural networks (done)

- [x] `nn.Module` system: parameters, buffers, children, train/eval, dtype
  conversion, named parameters, `state_dict`/`load_state_dict`
- [x] Linear, Conv2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Flatten, ReLU,
  LeakyReLU, Sigmoid, Tanh, Softmax, LogSoftmax, Dropout, BatchNorm2d, LayerNorm,
  Embedding
- [x] Sequential and ModuleList containers
- [x] Losses: MSE, L1, CrossEntropy, NLL, BCE, BCEWithLogits
- [x] 1-based class labels across `argmax` and classification losses
- [x] Optimizers: SGD (momentum, nesterov, weight decay), Adam, AdamW, RMSprop,
  Adagrad; gradient clipping; Step/MultiStep/Exponential/Cosine/Lambda schedulers
- [x] MLP and CNN training examples plus forward/backward/optimizer benchmarks
- [x] `torch.utils.data`: Dataset, TensorDataset, DataLoader, deterministic
  shuffling, `drop_last`
- [x] On-disk module serialization (`nn.save` / `model:load`, luatorch format)
- [x] PyTorch state-dict interchange through safetensors
- [x] GroupNorm
- [x] ConvTranspose2d
- [x] RNN and LSTM with multiple layers, bidirectionality, and dropout
- [x] Loss options: class weights, `ignore_index`, label smoothing
- [x] AMSGrad optimizer mode (fused)
- [x] Per-parameter optimizer groups
- [x] `ReduceLROnPlateau` scheduler

## v0.3 — tensor completeness (partial)

- [x] `gather`, `scatter`, `nonzero` (1-based coordinates), `index_select`
- [ ] Richer indexing: tensor-of-indices ranges, `index_put_`,
  `take_along_dim`, advanced assignment
- [x] `sort`, `topk`, `unique` (with inverse/counts)
- [ ] `bincount`, `histc`
- [x] Linear algebra: `inverse`, `det`, `solve`, `lstsq`, `cholesky`, `svd`, `qr`
- [x] `matrix_norm`, `vector_norm`, `cond`, `matrix_rank`, `slogdet`
- [x] `lu` with partial pivoting, batching, and autograd
- [x] `lu_factor`/`lu_solve` compact reusable solves with validation
- [x] `lu_unpack` rebuilds `P`, `L`, `U` from a packed factorization
- [x] `pinv` SVD-based pseudoinverse for rank-deficient and batched matrices
- [x] `matrix_exp` scaling-and-squaring matrix exponential with autograd
- [x] `diagonal`, `vander`, `cross`, and `multi_dot` real-valued helpers
- [x] `svdvals`, `vecdot`, `tensorinv`, `tensorsolve`, and
  `householder_product` real-valued `linalg.*` bindings
- [ ] `eig`, remaining `linalg.*` bindings
- [x] Random: `normal_`, `exponential_`, `bernoulli_`, `uniform_`, `multinomial`
- [ ] More distributions: `poisson`, `geometric`, `cauchy`, tensor-weighted
  `normal`
- [x] `mv`, `outer`, `addmm`, `baddbmm`
- [x] `einsum` (including ellipsis, repeated-label, and empty/scalar edges), `ger`
- [x] `logsumexp`
- [x] NaN/Inf predicates (`torch.isnan`, `torch.isinf`, `torch.isfinite`)
- [ ] More reductions: `aminmax`, `median`, `quantile`
- [x] Thread-count control (`torch.set_num_threads`)
- [x] `float16` creation from Lua values, `from_blob` buffers, and printing parity

## v0.4 — fused execution (done)

- [x] Trace elementwise functions into an expression tree
- [x] LuaJIT codegen backend for float32/float64
- [x] C codegen backend, `gcc -O3 -march=native -fopenmp`, on-disk kernel cache
- [x] `fused:into` output buffers; scalar arguments as runtime values
- [x] External-memory-aware GC (64 MB threshold) to fix out-of-place allocation
- [x] Fallback to eager for unsupported ops, broadcasting, and inference autograd
- [x] Head-to-head benchmark harness against PyTorch eager and `torch.compile`
- [x] 5.4x vs eager and ~1.0x vs `torch.compile` on `(a*b+c):relu()` (1M)
- [x] Fused reductions: `sum`, `mean` (one pass forward + backward)
- [x] Fused `logsumexp` (forward + backward)
- [x] Fused last-dimension `log_softmax` (forward + backward)
- [x] Fused cross-entropy (2D, mean/sum, class count <= 4096)
- [x] Fused last-dimension `softmax`
- [x] Fused softmax over arbitrary dimensions via layout move
- [x] First-order fused forward+backward through a C++ autograd Function
- [ ] Higher-order gradients through fused expressions
- [ ] Automatic tracing of ordinary operator code (lazy tensors)
- [ ] Linear/conv epilogue fusion (`x@W + b`, bias+activation)
- [ ] Vectorised transcendentals (libmvec or polynomial approximations)
- [x] Fused Adam/AdamW/SGD update kernels (5x optimizer step, 1.3x train step)
- [ ] Multi-output fusion and in-place fused kernels

## v0.5 — the Torch7 stack (current)

- [x] `torch.nn` layers, losses, containers, functional forms, parameter iteration
- [x] `torch.optim`: SGD, momentum, Adam, RMSprop, Adagrad
- [x] `torch.utils.data`: Dataset/DataLoader with deterministic shuffling
- [x] `torch.class`-style module system for user-defined layers (`nn.Module.class`)
- [x] Model serialization to disk (`nn.save` / `model:load`)
- [x] PyTorch state-dict interchange through safetensors
- [x] `luatorch` rockspec and system installation

## v0.6 — performance and ergonomics

- [ ] Zero-copy tensor views where safe; avoid redundant `contiguous()` calls
- [x] Direct `at::from_blob` wrappers for sharing memory with Lua strings/buffers, all dtypes
- [x] Batched index-select kernel
- [x] Fused optimizer updates (one C call per parameter instead of several ops)
- [x] Profiling helpers (`torch.profiler`, per-method timing/counts)
- [x] Improved large-tensor printing with edge previews
- [x] CI matrix (GCC/Clang on Linux; LuaJIT tests)

## v0.7 — profiling and release polish (done)

- [x] Opt-in per-tensor-method profiler with counts, total/average/max time
- [x] `0.7.0` version and release documentation
- [x] Direct `from_blob` wrappers for sharing Lua buffers
- [x] Batched indexing kernels
- [x] CI matrix

## v1.0 — GPU and distribution

- [ ] CUDA build via the CUDA libtorch distribution (`device = "cuda"`,
  `.cuda()`, `.cpu()`, device-aware constructors)
- [ ] CPU/GPU parity tests and device transfer benchmarks
- [ ] Optional MKL/OpenBLAS tuning and thread controls
- [ ] Stable C ABI versioning and prebuilt binary releases
- [ ] nn/optim GPU support

## Explicit non-goals

- Running unmodified Torch7 C extensions.
- Reimplementing `nn`/`optim` in Lua on top of a slow backend; modules should
  call ATen kernels directly.
- Windows support before the core is stable (patches welcome).

## How to help

Pick a v0.3 operator, a v0.2 layer, or a v0.4 fusion target. New fusion
features should include a correctness test, a fallback test, and a benchmark
entry in `bench/compare.py`. Benchmarks and compatibility notes are as valuable
as new ops.
