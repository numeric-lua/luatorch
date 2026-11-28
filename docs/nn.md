# Neural networks: `torch.nn` and `torch.optim`

Layers are plain Lua objects that call Tensor operations; gradients come from
ATen autograd, so there is no separate `backward()` method on modules. The
training loop is PyTorch's:

```lua
local torch = require("torch")
local nn = require("torch.nn")
local optim = require("torch.optim")

local model = nn.Sequential(
  nn.Linear(2, 32), nn.ReLU(),
  nn.Linear(32, 2))

local criterion = nn.CrossEntropyLoss()
local optimizer = optim.Adam(model:parameters(), { lr = 0.01 })

for epoch = 1, 100 do
  local logits = model(inputs)
  local loss = criterion(logits, labels)

  optimizer:zero_grad()
  loss:backward()
  optimizer:step()
end
```

`torch.nn` and `torch.optim` are also reachable as `torch.nn` / `torch.optim`
(lazy-loaded), but requiring them directly is the clearest style.

## Modules

Use `nn.Module` as the base class:

```lua
local nn = require("torch.nn")

local Scale = nn.Module.class("Scale")

function Scale.new(factor)
  local self = setmetatable(nn.Module.new(), Scale)
  self.factor = factor or 1
  return self
end

function Scale:forward(input)
  return input:mul(self.factor)
end
```

An instance is callable (`module(x)` calls `forward`) and supports:

| Method | Meaning |
| --- | --- |
| `module:forward(x)` | compute the output (autograd records the graph) |
| `module(x)` | same, via `__call` |
| `module:parameters()` | array of parameter tensors (recursive) |
| `module:named_parameters(prefix?)` | table `name -> tensor` |
| `module:named_buffers(prefix?)` | running statistics and other buffers |
| `module:zero_grad()` | zero all parameter gradients |
| `module:train(mode?)` / `module:eval()` | switch training mode (recursive) |
| `module:to(dtype)` | convert parameters/buffers, preserving `requires_grad` |
| `module:state_dict()` | snapshot `name -> tensor` (detached **clones**) |
| `module:load_state_dict(dict, strict?)` | copy values back; strict by default |
| `module:register_parameter(name, t)` | register a trainable tensor |
| `module:register_buffer(name, t)` | register a non-trainable tensor |
| `module:register_module(name, m)` | register a child module |

Parameters are registered with `requires_grad = true` (floating dtypes only).
In-place edits to parameters are rejected while they are leaves with gradients,
exactly like PyTorch; use `torch.no_grad(function() ... end)` for initialisation.

`state_dict` returns **clones**, not live views, so a snapshot is not corrupted
when training continues. This is slightly safer than PyTorch's shallow
references.

## Built-in modules

| Module | Signature |
| --- | --- |
| `nn.Linear(in, out, bias?)` | affine transform over the last dimension |
| `nn.Conv2d(in, out, kernel, opts?)` | 2-D convolution, NCHW, groups supported |
| `nn.MaxPool2d(kernel, stride?, padding?, dilation?, ceil_mode?)` | max pooling |
| `nn.AvgPool2d(kernel, stride?, padding?, ceil_mode?, count_include_pad?)` | average pooling |
| `nn.AdaptiveAvgPool2d(1)` | global average pooling |
| `nn.BatchNorm2d(features, opts?)` | `eps`, `momentum`, `affine`, `track_running_stats` |
| `nn.LayerNorm(normalized_shape, opts?)` | `eps`, `elementwise_affine`; normalizes trailing dims |
| `nn.GroupNorm(num_groups, channels, opts?)` | `eps`, `affine`; normalizes channel groups in NCHW |
| `nn.ConvTranspose2d(in, out, kernel, stride, padding, output_padding, groups, bias?)` | transposed convolution; output padding must be smaller than stride |
| `nn.Embedding(num_embeddings, dim)` | 1-based index lookup; `padding_idx` not supported yet |
| `nn.Dropout(p?, inplace?)` | inverted dropout |
| `nn.Dropout2d(p?, inplace?)` | elementwise dropout with a 2-D name |
| `nn.ReLU(inplace?)` | rectifier |
| `nn.LeakyReLU(slope?, inplace?)` | leaky rectifier |
| `nn.Sigmoid()` / `nn.Tanh()` | elementwise activations |
| `nn.Softplus(beta?, threshold?)` | default `beta = 1`, `threshold = 20` |
| `nn.ELU(alpha?, inplace?)` | default `alpha = 1` |
| `nn.SELU(inplace?)` | scaled exponential linear unit |
| `nn.SiLU(inplace?)` | `x * sigmoid(x)` |
| `nn.GELU(approximate?)` | exact `erf` form or `"tanh"` approximation |
| `nn.Softmax(dim?)` / `nn.LogSoftmax(dim?)` | default dim is the last one |
| `nn.Flatten(start_dim?, end_dim?)` | defaults `start_dim = 2`, `end_dim = -1` |
| `nn.Identity()` | pass-through |

Containers:

```lua
local model = nn.Sequential(
  nn.Conv2d(1, 8, 3, { padding = 1 }),
  nn.ReLU(),
  nn.MaxPool2d(2),
  nn.Flatten(),
  nn.Linear(8 * 4 * 4, 10))

model:add(nn.Dropout(0.2))     -- append a module
```

`nn.ModuleList({m1, m2})` stores modules for manual iteration
(`list:get(1)`, `#list:parameters()`).

`Conv2d` options can be a table (`{stride = 2, padding = 1, dilation = 1,
groups = 1, bias = true}`) or positional arguments. Kernel, stride, padding and
dilation accept a number or `{height, width}`.

## Data loading

`torch.utils.data` ships an in-memory dataset and a batcher. The full option,
iterator-lifecycle, worker, and limit reference is in [docs/data.md](data.md).
A quick example:

```lua
local data = require("torch.utils.data")

local dataset = data.TensorDataset(inputs, labels)   -- same first dimension
local loader = data.DataLoader(dataset, {
  batch_size = 64,
  shuffle = true,
  drop_last = false,
  num_workers = 2,      -- cooperative coroutine workers
  prefetch = 4,         -- prepared batches ahead of the current batch
  seed = 42,            -- optional deterministic shuffle
})

for batch_inputs, batch_labels in loader:iter() do
  local loss = criterion(model(batch_inputs), batch_labels)
  optimizer:zero_grad()
  loss:backward()
  optimizer:step()
end
```

- `dataset:size()` and `dataset:get(index)` (1-based) define a custom dataset.
- Datasets may implement `dataset:batch(indices)` to return a whole batch
  (TensorDataset uses `index_select`, avoiding per-sample overhead).
- Without `seed`, shuffling uses the global RNG; call `torch.manual_seed` for
  reproducibility. With `seed`, shuffling is deterministic per loader, does not
  consume the global RNG, and repeats the same order each epoch.
- `drop_last` discards an incomplete final batch; `loader:num_batches()` gives
  the number of batches per epoch.
- `num_workers` and `prefetch` do not change batch order. `prefetch` is bounded:
  at most `prefetch + 1` batches are held, and `0` disables look-ahead.
- There are no built-in image/text datasets yet; build tensors and wrap them.

Workers are Lua coroutines, not OS threads or processes. LuaJIT executes one Lua
coroutine at a time, so workers do not run in parallel with each other or with
training. A worker yields exactly one prepared batch per resume; this hides
batch preparation behind the loader's own control flow and avoids locks, copied
Lua states, and IPC, but a long native ATen call cannot overlap with worker
work. Iterators close automatically when an epoch is exhausted or a worker
fails. Call `iterator:close()` (or `loader:close()`) when leaving a loop early.
Worker errors are re-raised in batch order with the failing worker and batch in
the message.

## Losses

| Module | Notes |
| --- | --- |
| `nn.MSELoss(reduction?)` | mean over all elements by default |
| `nn.L1Loss(reduction?)` | |
| `nn.CrossEntropyLoss(options?)` | log-softmax + NLL; targets are **1-based**; supports `weight`, `ignore_index`, `label_smoothing` |
| `nn.NLLLoss(options?)` | input is already log-probabilities; supports `weight`, `ignore_index` |
| `nn.BCEWithLogitsLoss(reduction?)` | numerically stable binary cross entropy |
| `nn.BCELoss(reduction?)` | probabilities, clamped to avoid logs of zero |

`reduction` is `"mean"` (default), `"sum"` or `"none"`. The classification
losses accept `weight` (one value per class), `ignore_index` (a 1-based label to
skip), and `CrossEntropyLoss` also accepts `label_smoothing` in `[0, 1]`.
The weighted/ignore/smoothing path is implemented in Lua on 2-D logits and
log-probabilities; the default path keeps the fused C kernel.

`nn.Softmax` and `nn.LogSoftmax` use fused fast-math kernels (float32/float64)
for any dimension by moving that dimension last, making the layout contiguous,
and moving the result back. The default last-dimension path avoids the copies.
`CrossEntropyLoss` on 2D float32/64 logits with mean/sum reduction uses a fused
fast-math path when the class count is at most 4096; larger classifiers fall
back to ATen.

`MSELoss` and `L1Loss` use the fused one-pass reductions when input and target
have the same shape and float32/float64 dtype (and reduction is not `"none"`).
Broadcasting or unsupported layouts fall back to the ATen loss. This makes the
common regression losses trainable without materialising the elementwise error
tensor.

### Class labels are 1-based

`argmax` returns 1-based indices, and cross entropy expects 1-based labels:

```lua
local target = torch.LongTensor({1, 3, 2})
local loss = nn.CrossEntropyLoss()(logits, target)
local predicted = logits:argmax(2)          -- also 1-based
```

Internally the labels are shifted by one before calling ATen. This matches
Torch7's `ClassNLLCriterion` and keeps Lua code free of off-by-one conversions.

## Functional API

`nn.functional` exposes the stateless versions:

```lua
local F = require("torch.nn").functional

F.linear(x, weight, bias)
F.relu(x, inplace)
F.leaky_relu(x, slope, inplace)
F.sigmoid(x)   F.tanh(x)
F.softplus(x, beta, threshold)
F.elu(x, alpha, inplace)   F.selu(x, inplace)   F.silu(x, inplace)
F.gelu(x, approximate)
F.hardsigmoid(x, inplace)  F.hardswish(x, inplace)  F.softsign(x)
F.softmax(x, dim)   F.log_softmax(x, dim)
F.conv2d(x, weight, bias, stride, padding, dilation, groups)
F.max_pool2d(x, kernel, stride, padding, dilation, ceil_mode)
F.adaptive_avg_pool2d(x, 1)
F.mse_loss(x, target, reduction)
F.l1_loss(x, target, reduction)
F.cross_entropy(x, target, reduction)
F.nll_loss(x, target, reduction)
F.binary_cross_entropy_with_logits(x, target, reduction)
F.binary_cross_entropy(x, target, reduction)   -- probabilities
F.dropout(x, p, training, inplace)
```

## Recurrent layers

Multi-layer, optionally bidirectional RNN and LSTM layers are available:

```lua
local rnn = nn.RNN(16, 32, { batch_first = true, nonlinearity = "tanh" })
local lstm = nn.LSTM(16, 32, { batch_first = true })

local output, hidden = rnn(input)
local output2, hidden2, cell2 = lstm(input)
```

- Input is `[seq, batch, input]` by default or `[batch, seq, input]` with
  `batch_first = true`; a 2-D input is treated as one unbatched sequence.
- `num_layers` stacks cells; `bidirectional = true` runs forward and reverse
  passes and concatenates their outputs.
- `dropout` is applied between layers in training mode and disabled in `eval()`.
- Hidden/cell states are optional; pass `[num_layers * directions, batch,
  hidden]` tensors (or `[batch, hidden]` for a single direction) to continue a
  sequence.
- Parameters use PyTorch names, including `_l1` and `_reverse` suffixes, so
  safetensors state dictionaries map directly.

## Optimizers

All optimizers take `(parameters, options)` and expose `step()` and
`zero_grad()`. Updates run under `torch.no_grad`.

Parameters can also be supplied as PyTorch-style groups:

```lua
local optimizer = optim.Adam({
  { params = encoder:parameters(), lr = 1e-4 },
  { params = head:parameters(), lr = 1e-3, weight_decay = 0.01 },
}, { lr = 1e-3 })

optimizer:add_param_group({ params = extra:parameters(), lr = 5e-4 })
```

Each group carries its own `lr` and optimizer options; `optimizer.lr` mirrors
the first group. Learning-rate schedulers update every group independently.

| Optimizer | Options |
| --- | --- |
| `optim.SGD` | `lr`, `momentum`, `dampening`, `weight_decay`, `nesterov` |
| `optim.Adam` | `lr`, `betas = {b1, b2}`, `eps`, `weight_decay`, `amsgrad` |
| `optim.AdamW` | Adam with decoupled weight decay and optional `amsgrad` |
| `optim.RMSprop` | `lr`, `alpha`, `eps`, `weight_decay`, `momentum`, `centered` |
| `optim.Adagrad` | `lr`, `lr_decay`, `weight_decay`, `eps`, `initial_accumulator_value` |

Passing a number as options means `{lr = number}`. Per-parameter state is keyed
by tensor identity.

### Linear

`nn.Linear` uses a single `addmm` for 2D inputs, folding the bias into the
matrix multiply. Higher-rank inputs are flattened to 2D, projected, and
reshaped back. This avoids a separate bias kernel and temporary.

### Fused update kernels

Adam, AdamW, SGD, RMSprop and Adagrad execute their update in a single C kernel
over contiguous float32/float64 storage, instead of a chain of temporary tensor
ops. Adam/AdamW with `amsgrad = true` use the fused AMSGrad kernel and track the
maximum second moment per parameter. On a 784→256→10 MLP this reduces the Adam step from ~1.14 ms (PyTorch) to
~0.22 ms, and the complete training step from ~5.1 ms to ~4.0 ms.

Parameters, gradients and optimizer state must be contiguous and float32 or
float64; non-contiguous parameters raise an error rather than silently copying.

### Gradient clipping

Gradient-norm and gradient-value clipping use fused C kernels when every
gradient is contiguous float32/float64; other layouts fall back to the Lua
tensor path.

```lua
optim.clip_grad_norm_(model:parameters(), 1.0)        -- global L2 norm
optim.clip_grad_norm_(model:parameters(), 1.0, math.huge)
optim.clip_grad_value_(model:parameters(), 0.5)       -- elementwise clamp
```

### Learning-rate schedulers

Call `scheduler:step()` once per epoch, after `optimizer:step()`:

```lua
local scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
local scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, {30, 60}, 0.1)
local scheduler3 = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
local scheduler4 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)
local scheduler5 = optim.lr_scheduler.LambdaLR(optimizer, function(epoch)
  return 1 / epoch
end)
local scheduler6 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, {
  mode = "min", factor = 0.5, patience = 5, min_lr = 1e-6,
})
```

Schedulers update every parameter group; they do not own parameter state.
`ReduceLROnPlateau:step(metric)` takes the validation metric and reduces the
rate after `patience` non-improving epochs, with optional cooldown and `min_lr`
clamping.

Create the optimizer **after** converting the model dtype, because
`module:to(dtype)` replaces parameter objects:

```lua
model:to(torch.float64)
local optimizer = optim.Adam(model:parameters(), { lr = 1e-3 })
```

## Saving and loading models

```lua
model:save("classifier.luatorch")
model:load("classifier.luatorch")

-- functional equivalents
nn.save(model, "classifier.luatorch")
nn.load(model, "classifier.luatorch")
```

The file contains the state dictionary (parameters plus buffers) in a compact
luatorch-specific format: a small text header per tensor followed by contiguous
bytes. Loading requires the same architecture; values are copied with
`load_state_dict`, so dtypes are cast when needed.

For interchange with the PyTorch ecosystem, use the safetensors format:

```lua
model:save_safetensors("classifier.safetensors")
model:load_safetensors("classifier.safetensors")
model:load_safetensors("classifier.safetensors",
  { strip_prefix = "model.", strict = true })
```

State-dict keys are the module parameter/buffer names. A PyTorch model can be
written with `safetensors.torch.save_file(model.state_dict(), path)` and loaded
here directly; the reverse works with `safetensors.torch.load_file`. Optional
`prefix`/`strip_prefix` handle wrapper names such as `model.`. Float16 tensors
are written but cannot currently be loaded (`from_blob` limitation).

## Training recipe

```lua
local loader = data.DataLoader(dataset, {batch_size = 64, shuffle = true})
local scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)

for epoch = 1, epochs do
  model:train()
  for inputs, targets in loader:iter() do
    local loss = criterion(model(inputs), targets)
    optimizer:zero_grad()
    loss:backward()
    optimizer:step()
  end
  scheduler:step()
end

model:eval()
local accuracy = model(test_inputs):argmax(2):eq(test_targets)
  :to(torch.float32):mean():item()
```

- Call `zero_grad()` before `backward()` each iteration.
- Wrap evaluation in `torch.no_grad(function() ... end)` if you want to skip
  graph construction.
- `BatchNorm2d` uses batch statistics in training mode (updating running
  estimates) and running statistics in `eval()`.
- `Dropout` is identity in `eval()` and scales survivors by `1/(1-p)` in
  training.

## Current limitations

- No built-in datasets; `DataLoader` works on in-memory tensors and custom
  dataset objects.
- `nn.save`/`nn.load` use a luatorch-specific format; safetensors provides
  PyTorch state-dict interchange.
- RNN/LSTM are single-layer and unidirectional; weight-norm layers are not
  implemented.
- `Embedding` does not support `padding_idx` yet.
- Weighted/ignore-index classification losses currently require 2-D logits.
- Modules are CPU-only until the CUDA libtorch build lands.
