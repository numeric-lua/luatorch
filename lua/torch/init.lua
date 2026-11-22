--
-- luatorch / init.lua
--
-- Public entry point.  Mirrors the ergonomics of both PyTorch and Torch7:
--
--   local torch = require("torch")
--   local x = torch.randn(2, 3)
--   local y = x:matmul(torch.ones(3, 4))
--   print(y)
--
-- Functional utilities come from luafun (fun.lua) instead of Moses:
--
--   local fun = require("fun")          -- or torch.fun
--   fun.iter({1, 2, 3}):map(function(x) return x * x end):totable()
--
local libmod = require("torch.lib")
local ffi = libmod.ffi
local lib = libmod.lib
local Tensor = require("torch.tensor")
local fuse = require("torch.fuse")
local fun = require("fun")

local torch = {}

torch.VERSION = "0.22.0"
torch.version = libmod.version
torch.__version = libmod.version
torch.Tensor = Tensor
torch.is_tensor = Tensor.is_tensor
torch.from_blob = Tensor.from_blob
torch.profiler = require("torch.profiler")
torch.fuse = fuse
torch.fuse_train = fuse.train
torch.fuse_reduce = fuse.reduce
torch.ALL_DIMS = Tensor.ALL_DIMS
torch.fun = fun
torch.luafun = fun

-- dtype ids (also accepted as strings everywhere) ---------------------------
torch.float32 = 0
torch.float = 0
torch.float64 = 1
torch.double = 1
torch.int64 = 2
torch.long = 2
torch.int32 = 3
torch.int = 3
torch.int16 = 4
torch.short = 4
torch.int8 = 5
torch.char = 5
torch.uint8 = 6
torch.byte = 6
torch.bool = 7
torch.boolean = 7
torch.float16 = 8
torch.half = 8

local default_dtype = 0

local function with_default(opts)
  opts = opts or {}
  if opts.dtype ~= nil then
    return opts
  end
  local copy = {}
  for key, value in pairs(opts) do
    copy[key] = value
  end
  copy.dtype = default_dtype
  return copy
end

-- creation ------------------------------------------------------------------

function torch.tensor(value, opts)
  opts = opts or {}
  if Tensor.is_tensor(value) then
    return value:clone()
  end
  if type(value) == "table" then
    return Tensor.from_table(value, opts)
  end
  if type(value) == "number" or type(value) == "boolean" then
    return Tensor.from_scalar(value, opts)
  end
  error("torch.tensor expects a table, number, boolean or Tensor", 2)
end

local function size_creator(kind)
  return function(...)
    local shape, opts = Tensor._parse_shape_opts(...)
    return Tensor._create(kind, shape, with_default(opts))
  end
end

torch.zeros = size_creator("zeros")
torch.ones = size_creator("ones")
torch.empty = size_creator("empty")
torch.rand = size_creator("rand")
torch.randn = size_creator("randn")

function torch.full(...)
  local args = { ... }
  local opts = {}
  if #args > 0 and Tensor._is_options(args[#args]) then
    opts = table.remove(args)
  end
  local value = table.remove(args)
  if type(value) ~= "number" then
    error("torch.full expects a numeric fill value", 2)
  end
  local shape
  if #args == 1 and type(args[1]) == "table" then
    shape = args[1]
  else
    shape = args
  end
  if #shape == 0 then
    error("torch.full expects a tensor shape", 2)
  end
  return Tensor._create("full", shape, with_default(opts), value)
end

function torch.eye(n, opts)
  opts = with_default(opts)
  return Tensor._wrap(lib.lt_eye(n, Tensor.resolve_dtype(opts.dtype)))
end

function torch.arange(...)
  local args = { ... }
  local opts = {}
  if #args > 0 and Tensor._is_options(args[#args]) then
    opts = table.remove(args)
  end
  if #args < 1 or #args > 3 then
    error("torch.arange expects start, stop[, step]", 2)
  end
  local start, stop, step
  if #args == 1 then
    start, stop, step = 0, args[1], 1
  elseif #args == 2 then
    start, stop, step = args[1], args[2], 1
  else
    start, stop, step = args[1], args[2], args[3]
  end
  local dtype = opts.dtype
  if dtype == nil then
    dtype = (start % 1 == 0 and stop % 1 == 0 and step % 1 == 0) and 2 or 0
  end
  return Tensor._wrap(lib.lt_arange(start, stop, step, Tensor.resolve_dtype(dtype)))
end

function torch.linspace(start, stop, steps, opts)
  steps = steps or 100
  opts = with_default(opts)
  return Tensor._wrap(lib.lt_linspace(start, stop, steps, Tensor.resolve_dtype(opts.dtype)))
end

function torch.randperm(n, opts)
  opts = with_default(opts)
  return Tensor._wrap(lib.lt_randperm(n, Tensor.resolve_dtype(opts.dtype)))
end

-- like-constructors ----------------------------------------------------------

function torch.zeros_like(t, opts)
  local out = t:detach():clone()
  if opts and opts.dtype then
    out = out:to(opts.dtype)
  end
  return out:zero_()
end

function torch.ones_like(t, opts)
  local function vector_values(input, name)
  if not Tensor.is_tensor(input) then input = torch.tensor(input) end
  if input:dim() ~= 1 then error(name .. " expects a 1-D tensor", 3) end
  return input
end

function torch.bincount(input, weights, minlength)
  input = vector_values(input, "bincount")
  local values = input:to(torch.int64):totable()
  local weight_values
  if weights ~= nil then
    weights = vector_values(weights, "bincount weights")
    if weights:numel() ~= #values then
      error("bincount weights length mismatch", 3)
    end
    weight_values = weights:to(torch.float64):totable()
  end
  if minlength ~= nil and (type(minlength) ~= "number" or minlength < 0 or
      minlength % 1 ~= 0) then
    error("bincount minlength must be a non-negative integer", 3)
  end
  local maximum = -1
  for _, value in ipairs(values) do
    if value < 0 then error("bincount input must be non-negative", 3) end
    if value > maximum then maximum = value end
  end
  local length = math.max(maximum + 1, minlength or 0)
  local counts = {}
  for i = 1, length do counts[i] = 0 end
  for i, value in ipairs(values) do
    counts[value + 1] = counts[value + 1] +
      (weight_values and weight_values[i] or 1)
  end
  return torch.tensor(counts, { dtype = weights and torch.float64 or torch.int64 })
end

function torch.histc(input, bins, min, max)
  input = vector_values(input, "histc")
  bins = bins or 100
  if type(bins) ~= "number" or bins < 1 or bins % 1 ~= 0 then
    error("histc bins must be a positive integer", 3)
  end
  local values = input:to(torch.float64):totable()
  if #values == 0 then return torch.zeros(bins, { dtype = torch.int64 }) end
  min = min or math.min(table.unpack(values))
  max = max or math.max(table.unpack(values))
  if min > max then error("histc min must not exceed max", 3) end
  local counts = {}
  for i = 1, bins do counts[i] = 0 end
  if min == max then
    for _, value in ipairs(values) do
      if value == min then counts[1] = counts[1] + 1 end
    end
  else
    local width = (max - min) / bins
    for _, value in ipairs(values) do
      if value >= min and value <= max then
        local index = math.floor((value - min) / width) + 1
        if index > bins then index = bins end
        counts[index] = counts[index] + 1
      end
    end
  end
  return torch.tensor(counts, { dtype = torch.int64 })
end

return torch.zeros_like(t, opts):fill_(1)
end

function torch.full_like(t, value, opts)
  local function vector_values(input, name)
  if not Tensor.is_tensor(input) then input = torch.tensor(input) end
  if input:dim() ~= 1 then error(name .. " expects a 1-D tensor", 3) end
  return input
end

function torch.bincount(input, weights, minlength)
  input = vector_values(input, "bincount")
  local values = input:to(torch.int64):totable()
  local weight_values
  if weights ~= nil then
    weights = vector_values(weights, "bincount weights")
    if weights:numel() ~= #values then
      error("bincount weights length mismatch", 3)
    end
    weight_values = weights:to(torch.float64):totable()
  end
  if minlength ~= nil and (type(minlength) ~= "number" or minlength < 0 or
      minlength % 1 ~= 0) then
    error("bincount minlength must be a non-negative integer", 3)
  end
  local maximum = -1
  for _, value in ipairs(values) do
    if value < 0 then error("bincount input must be non-negative", 3) end
    if value > maximum then maximum = value end
  end
  local length = math.max(maximum + 1, minlength or 0)
  local counts = {}
  for i = 1, length do counts[i] = 0 end
  for i, value in ipairs(values) do
    counts[value + 1] = counts[value + 1] +
      (weight_values and weight_values[i] or 1)
  end
  return torch.tensor(counts, { dtype = weights and torch.float64 or torch.int64 })
end

function torch.histc(input, bins, min, max)
  input = vector_values(input, "histc")
  bins = bins or 100
  if type(bins) ~= "number" or bins < 1 or bins % 1 ~= 0 then
    error("histc bins must be a positive integer", 3)
  end
  local values = input:to(torch.float64):totable()
  if #values == 0 then return torch.zeros(bins, { dtype = torch.int64 }) end
  min = min or math.min(table.unpack(values))
  max = max or math.max(table.unpack(values))
  if min > max then error("histc min must not exceed max", 3) end
  local counts = {}
  for i = 1, bins do counts[i] = 0 end
  if min == max then
    for _, value in ipairs(values) do
      if value == min then counts[1] = counts[1] + 1 end
    end
  else
    local width = (max - min) / bins
    for _, value in ipairs(values) do
      if value >= min and value <= max then
        local index = math.floor((value - min) / width) + 1
        if index > bins then index = bins end
        counts[index] = counts[index] + 1
      end
    end
  end
  return torch.tensor(counts, { dtype = torch.int64 })
end

return torch.zeros_like(t, opts):fill_(value)
end

local function random_like(t, opts, generator)
  opts = opts or {}
  local dtype = opts.dtype or t:dtype()
  local sizes = t:sizes()
  if #sizes == 0 then
    return generator(1, { dtype = dtype }):reshape({})
  end
  return generator(sizes, { dtype = dtype })
end

function torch.rand_like(t, opts)
  return random_like(t, opts, torch.rand)
end

function torch.randn_like(t, opts)
  return random_like(t, opts, torch.randn)
end

-- joining --------------------------------------------------------------------

local function tensor_array(tensors, what)
  if type(tensors) ~= "table" or #tensors == 0 then
    error(what .. " expects a non-empty table of tensors", 3)
  end
  local n = #tensors
  local array = ffi.new("lt_tensor*[?]", n)
  for i = 1, n do
    local t = tensors[i]
    if not Tensor.is_tensor(t) then
      error(what .. " expects tensors, got " .. type(t) .. " at position " .. i, 3)
    end
    array[i - 1] = t.h
  end
  return array, n
end

function torch.cat(tensors, dim)
  local array, n = tensor_array(tensors, "torch.cat")
  dim = dim or 1
  local cdim = dim > 0 and dim - 1 or dim
  return Tensor._wrap(lib.lt_cat(array, n, cdim))
end

function torch.stack(tensors, dim)
  local array, n = tensor_array(tensors, "torch.stack")
  dim = dim or 1
  local cdim = dim > 0 and dim - 1 or dim
  return Tensor._wrap(lib.lt_stack(array, n, cdim))
end

function torch.where(condition, a, b)
  if not (Tensor.is_tensor(condition) and Tensor.is_tensor(a) and Tensor.is_tensor(b)) then
    error("torch.where expects three tensors", 2)
  end
  return Tensor._wrap(lib.lt_where(condition.h, a.h, b.h))
end

-- Torch7-style typed constructors -------------------------------------------

local function typed_constructor(dtype)
  return function(...)
    local args = { ... }
    if #args == 1 and type(args[1]) == "table" then
      return Tensor.from_table(args[1], { dtype = dtype })
    end
    -- A single number is a length, not a scalar (Torch7 behaviour).
    local shape = Tensor._parse_shape_opts(...)
    return Tensor._create("empty", shape, { dtype = dtype })
  end
end

torch.FloatTensor = typed_constructor(0)
torch.DoubleTensor = typed_constructor(1)
torch.LongTensor = typed_constructor(2)
torch.IntTensor = typed_constructor(3)
torch.ShortTensor = typed_constructor(4)
torch.CharTensor = typed_constructor(5)
torch.ByteTensor = typed_constructor(6)
torch.BoolTensor = typed_constructor(7)
torch.HalfTensor = typed_constructor(8)

-- autograd context -----------------------------------------------------------

function torch.set_grad_enabled(enabled)
  lib.lt_set_grad_enabled(enabled and 1 or 0)
end

function torch.is_grad_enabled()
  return lib.lt_is_grad_enabled() == 1
end

function torch.no_grad(fn)
  local previous = lib.lt_is_grad_enabled()
  lib.lt_set_grad_enabled(0)
  local ok, result = pcall(fn)
  lib.lt_set_grad_enabled(previous)
  if not ok then
    error(result, 0)
  end
  return result
end

-- free-function aliases ------------------------------------------------------

torch.add = function(a, b) return a:add(b) end
torch.sub = function(a, b) return a:sub(b) end
torch.mul = function(a, b) return a:mul(b) end
torch.div = function(a, b) return a:div(b) end
torch.pow = function(a, b) return a:pow(b) end
torch.matmul = function(a, b) return a:matmul(b) end
torch.mm = function(a, b) return a:mm(b) end
torch.bmm = function(a, b) return a:bmm(b) end
torch.dot = function(a, b) return a:dot(b) end
torch.sum = function(t, dim, keepdim) return t:sum(dim, keepdim) end
torch.mean = function(t, dim, keepdim) return t:mean(dim, keepdim) end
torch.prod = function(t, dim, keepdim) return t:prod(dim, keepdim) end
torch.max = function(t, dim, keepdim) return t:max(dim, keepdim) end
torch.min = function(t, dim, keepdim) return t:min(dim, keepdim) end
torch.argmax = function(t, dim, keepdim) return t:argmax(dim, keepdim) end
torch.argmin = function(t, dim, keepdim) return t:argmin(dim, keepdim) end
torch.transpose = function(t, d1, d2) return t:transpose(d1, d2) end
torch.reshape = function(t, ...) return t:reshape(...) end
torch.clamp = function(t, lo, hi) return t:clamp(lo, hi) end
torch.relu = function(t) return t:relu() end
torch.sigmoid = function(t) return t:sigmoid() end
torch.tanh = function(t) return t:tanh() end
torch.exp = function(t) return t:exp() end
torch.log = function(t) return t:log() end
torch.sqrt = function(t) return t:sqrt() end
torch.abs = function(t) return t:abs() end

-- Trapezoidal integration. `x` may be a tensor of sample positions; when it
-- is omitted, `dx` (default 1) is the uniform spacing. `opts` accepts `dim`
-- and `dx`, and the integrated dimension defaults to the last one.
function torch.trapezoid(y, x, opts)
  if not Tensor.is_tensor(y) then
    error("torch.trapezoid expects a tensor as its first argument", 2)
  end
  if x ~= nil and not Tensor.is_tensor(x) then
    if type(x) == "table" and opts == nil then
      opts, x = x, nil
    else
      error("torch.trapezoid x must be a Tensor or nil", 2)
    end
  end
  return y:trapezoid(x, opts)
end

function torch.equal(a, b)
  return a:equal(b)
end

function torch.allclose(a, b, rtol, atol)
  return a:allclose(b, rtol, atol)
end

-- NaN / Inf predicates -------------------------------------------------------

function torch.isnan(t)
  if not Tensor.is_tensor(t) then
    error("torch.isnan expects a tensor", 2)
  end
  return t:isnan()
end

function torch.isinf(t)
  if not Tensor.is_tensor(t) then
    error("torch.isinf expects a tensor", 2)
  end
  return t:isinf()
end

function torch.isfinite(t)
  if not Tensor.is_tensor(t) then
    error("torch.isfinite expects a tensor", 2)
  end
  return t:isfinite()
end

-- neural-network primitives -------------------------------------------------

local function as_pair(value, name)
  if value == nil then
    return 1, 1
  end
  if type(value) == "number" then
    return value, value
  end
  if type(value) == "table" and #value == 2 then
    return value[1], value[2]
  end
  error(name .. " must be a number or a {height, width} table", 3)
end

local reduction_codes = { none = 0, mean = 1, sum = 2 }

local function reduction_code(value)
  if value == nil then
    return 1
  end
  if type(value) == "number" then
    if value >= 0 and value <= 2 then
      return value
    end
    error("reduction must be 0 (none), 1 (mean) or 2 (sum)", 3)
  end
  local code = reduction_codes[value]
  if code == nil then
    error("unknown reduction '" .. tostring(value) .. "'", 3)
  end
  return code
end

function torch.softmax(input, dim)
  return input:softmax(dim)
end

function torch.log_softmax(input, dim)
  return input:log_softmax(dim)
end

function torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
  if not Tensor.is_tensor(input) or not Tensor.is_tensor(weight) then
    error("torch.conv2d expects input and weight tensors", 2)
  end
  local stride_h, stride_w = as_pair(stride or 1, "stride")
  local pad_h, pad_w = as_pair(padding or 0, "padding")
  local dilation_h, dilation_w = as_pair(dilation or 1, "dilation")
  local bias_pointer = nil
  if bias ~= nil and bias ~= false then
    if not Tensor.is_tensor(bias) then
      error("torch.conv2d: bias must be a Tensor or nil", 2)
    end
    bias_pointer = bias.h
  end
  return Tensor._wrap(lib.lt_conv2d(
    input.h, weight.h, bias_pointer,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups or 1))
end

function torch.conv_transpose2d(input, weight, bias, stride, padding,
    output_padding, groups, dilation)
  if not Tensor.is_tensor(input) or not Tensor.is_tensor(weight) then
    error("torch.conv_transpose2d expects input and weight tensors", 2)
  end
  local stride_h, stride_w = as_pair(stride or 1, "stride")
  local pad_h, pad_w = as_pair(padding or 0, "padding")
  local output_pad_h, output_pad_w = as_pair(output_padding or 0,
    "output_padding")
  local dilation_h, dilation_w = as_pair(dilation or 1, "dilation")
  local bias_pointer = nil
  if bias ~= nil and bias ~= false then
    if not Tensor.is_tensor(bias) then
      error("torch.conv_transpose2d: bias must be a Tensor or nil", 2)
    end
    bias_pointer = bias.h
  end
  return Tensor._wrap(lib.lt_conv_transpose2d(
    input.h, weight.h, bias_pointer,
    stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w,
    dilation_h, dilation_w, groups or 1))
end

function torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
  if not Tensor.is_tensor(input) then
    error("torch.max_pool2d expects a Tensor", 2)
  end
  local kernel_h, kernel_w = as_pair(kernel_size, "kernel_size")
  local stride_h, stride_w = as_pair(stride or kernel_size, "stride")
  local pad_h, pad_w = as_pair(padding or 0, "padding")
  local dilation_h, dilation_w = as_pair(dilation or 1, "dilation")
  return Tensor._wrap(lib.lt_max_pool2d(
    input.h,
    kernel_h, kernel_w, stride_h, stride_w,
    pad_h, pad_w, dilation_h, dilation_w,
    ceil_mode and 1 or 0))
end

function torch.avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
  if not Tensor.is_tensor(input) then
    error("torch.avg_pool2d expects a Tensor", 2)
  end
  local kernel_h, kernel_w = as_pair(kernel_size, "kernel_size")
  local stride_h, stride_w = as_pair(stride or kernel_size, "stride")
  local pad_h, pad_w = as_pair(padding or 0, "padding")
  if count_include_pad == nil then
    count_include_pad = true
  end
  return Tensor._wrap(lib.lt_avg_pool2d(
    input.h, kernel_h, kernel_w, stride_h, stride_w,
    pad_h, pad_w, ceil_mode and 1 or 0, count_include_pad and 1 or 0))
end

local function shape_cdata(shape)
  local array = ffi.new("int64_t[?]", #shape)
  for i = 1, #shape do
    array[i - 1] = shape[i]
  end
  return array, #shape
end

function torch.embedding(weight, indices)
  if not Tensor.is_tensor(weight) or not Tensor.is_tensor(indices) then
    error("torch.embedding expects weight and indices tensors", 2)
  end
  -- Embedding indices are 1-based at the Lua surface.
  local zero_based = indices:to(torch.int64):sub(1)
  return Tensor._wrap(lib.lt_embedding(weight.h, zero_based.h))
end

function torch.layer_norm(input, normalized_shape, weight, bias, eps)
  if not Tensor.is_tensor(input) then
    error("torch.layer_norm expects a Tensor input", 2)
  end
  local shape
  if type(normalized_shape) == "number" then
    shape = { normalized_shape }
  elseif type(normalized_shape) == "table" then
    shape = normalized_shape
  else
    error("torch.layer_norm: normalized_shape must be a number or table", 2)
  end
  local weight_pointer = nil
  if weight ~= nil then
    if not Tensor.is_tensor(weight) then
      error("torch.layer_norm: weight must be a Tensor or nil", 2)
    end
    weight_pointer = weight.h
  end
  local bias_pointer = nil
  if bias ~= nil then
    if not Tensor.is_tensor(bias) then
      error("torch.layer_norm: bias must be a Tensor or nil", 2)
    end
    bias_pointer = bias.h
  end
  local array, ndim = shape_cdata(shape)
  return Tensor._wrap(lib.lt_layer_norm(
    input.h, weight_pointer, bias_pointer, ndim, array, eps or 1e-5))
end

function torch.cross_entropy(input, target, reduction)
  if not Tensor.is_tensor(input) or not Tensor.is_tensor(target) then
    error("torch.cross_entropy expects input and target tensors", 2)
  end
  -- Class labels are 1-based at the Lua surface; ATen expects 0-based targets.
  local zero_based = target:to(torch.int64):sub(1)
  local code = reduction_code(reduction)
  local dtype = input:dtype_id()
  local sizes = input:sizes()
  local class_count = #sizes >= 2 and sizes[2] or 0
  if input:dim() == 2 and target:dim() == 1 and
    (code == 1 or code == 2) and (dtype == 0 or dtype == 1) and
    class_count <= 4096 then
    local fused = lib.lt_fused_cross_entropy(input.h, zero_based.h, code)
    if fused ~= nil then
      return Tensor._wrap(fused)
    end
  end
  return Tensor._wrap(lib.lt_cross_entropy(input.h, zero_based.h, code))
end

function torch.nll_loss(input, target, reduction)
  if not Tensor.is_tensor(input) or not Tensor.is_tensor(target) then
    error("torch.nll_loss expects input and target tensors", 2)
  end
  local zero_based = target:to(torch.int64):sub(1)
  return Tensor._wrap(lib.lt_nll_loss(input.h, zero_based.h, reduction_code(reduction)))
end

function torch.binary_cross_entropy_with_logits(input, target, reduction)
  if not Tensor.is_tensor(input) or not Tensor.is_tensor(target) then
    error("torch.binary_cross_entropy_with_logits expects two tensors", 2)
  end
  return Tensor._wrap(lib.lt_binary_cross_entropy_with_logits(
    input.h, target.h, reduction_code(reduction)))
end

function torch.uniform_(tensor, low, high)
  if not Tensor.is_tensor(tensor) then
    error("torch.uniform_ expects a Tensor", 2)
  end
  return tensor:uniform_(low, high)
end

-- sorting, selection, gather/scatter ----------------------------------------

function torch.sort(tensor, dim, descending)
  return tensor:sort(dim, descending)
end

function torch.topk(tensor, k, dim, largest, sorted_flag)
  return tensor:topk(k, dim, largest, sorted_flag)
end

function torch.nonzero(tensor)
  return tensor:nonzero()
end

function torch.gather(tensor, dim, index)
  return tensor:gather(dim, index)
end

function torch.scatter(tensor, dim, index, source)
  return tensor:scatter(dim, index, source)
end

function torch.unique(tensor, sorted_flag, return_inverse, return_counts)
  return tensor:unique(sorted_flag, return_inverse, return_counts)
end

function torch.multinomial(tensor, num_samples, replacement)
  return tensor:multinomial(num_samples, replacement)
end

-- linear algebra -------------------------------------------------------------

torch.linalg = {
  inv = function(a) return a:inverse() end,
  det = function(a) return a:det() end,
  slogdet = function(a) return a:slogdet() end,
  solve = function(a, b) return a:solve(b) end,
  solve_triangular = function(a, b, opts)
    return a:solve_triangular(b, opts)
  end,
  lstsq = function(a, b, opts) return a:lstsq(b, opts) end,
  cholesky = function(a, upper) return a:cholesky(upper) end,
  svd = function(a, full_matrices) return a:svd(full_matrices) end,
  qr = function(a, mode) return a:qr(mode) end,
  matrix_power = function(a, n) return a:matrix_power(n) end,
  vector_norm = function(a, ord, dim, keepdim)
    if not Tensor.is_tensor(a) then
      error("vector_norm expects a tensor", 3)
    end
    ord = ord == nil and 2 or ord
    if type(ord) ~= "number" then
      error("vector_norm ord must be a number", 3)
    end
    return a:norm(ord, dim, keepdim)
  end,
  cond = function(a, p)
    if not Tensor.is_tensor(a) or a:dim() ~= 2 then
      error("cond expects a 2-D tensor", 3)
    end
    p = p == nil and 2 or p
    local _, singular = a:svd()
    if p == 2 then return singular:max():item() / singular:min():item() end
    local inverse = a:inverse()
    if p == 1 then
      return torch.linalg.matrix_norm(a, 1) * torch.linalg.matrix_norm(inverse, 1)
    elseif p == math.huge then
      return torch.linalg.matrix_norm(a, math.huge) *
        torch.linalg.matrix_norm(inverse, math.huge)
    elseif p == "fro" then
      return torch.linalg.matrix_norm(a, "fro") *
        torch.linalg.matrix_norm(inverse, "fro")
    end
    error("cond supports 2, 1, inf, and 'fro'", 3)
  end,
  matrix_rank = function(a, tol)
    if not Tensor.is_tensor(a) or a:dim() ~= 2 then
      error("matrix_rank expects a 2-D tensor", 3)
    end
    local _, singular = a:svd()
    local values = singular:totable()
    if tol == nil then
      local largest = 0
      for _, value in ipairs(values) do
        if value > largest then largest = value end
      end
      tol = math.max(a:size(1), a:size(2)) * 2.220446049250313e-16 * largest
    end
    local rank = 0
    for _, value in ipairs(values) do
      if value > tol then rank = rank + 1 end
    end
    return rank
  end,
  matrix_norm = function(a, ord)
    if not Tensor.is_tensor(a) or a:dim() ~= 2 then
      error("matrix_norm expects a 2-D tensor", 3)
    end
    if ord == nil or ord == "fro" then
      return a:norm(2):item()
    end
    local row_sums = a:abs():sum(2)
    local column_sums = a:abs():sum(1)
    if ord == 2 or ord == -2 then
      local _, singular_values = a:svd()
      if ord == 2 then return singular_values:max():item() end
      return singular_values:min():item()
    end
    if ord == 1 then return column_sums:max():item() end
    if ord == -1 then return column_sums:min():item() end
    if ord == math.huge then return row_sums:max():item() end
    if ord == -math.huge then return row_sums:min():item() end
    error("matrix_norm supports 'fro', 2, -2, 1, -1, inf, and -inf", 3)
  end,
}

function torch.mv(matrix, vector)
  return matrix:mv(vector)
end

function torch.outer(a, b)
  return a:outer(b)
end

function torch.ger(a, b)
  a = Tensor.is_tensor(a) and a or torch.tensor(a)
  b = Tensor.is_tensor(b) and b or torch.tensor(b)
  if a:dim() ~= 1 or b:dim() ~= 1 then
    error("ger expects two 1-D tensors", 2)
  end
  return a:outer(b)
end

-- Einstein summation ---------------------------------------------------------

-- ATen parses the equation again and remains the authority on rank/shape
-- errors; this pre-pass rejects malformed equations and operand-count
-- mismatches with Lua-level messages before crossing the FFI boundary.
local function check_einsum_subscript(subscript, what)
  local i, ellipses = 1, 0
  while i <= #subscript do
    local ch = subscript:sub(i, i)
    if ch == "." then
      if subscript:sub(i, i + 2) ~= "..." then
        error("einsum: invalid '.' in " .. what ..
          " (use '...' for an ellipsis)", 3)
      end
      ellipses = ellipses + 1
      if ellipses > 1 then
        error("einsum: more than one ellipsis in " .. what, 3)
      end
      i = i + 3
    elseif ch:match("[a-zA-Z]") then
      i = i + 1
    else
      error("einsum: invalid character '" .. ch .. "' in " .. what ..
        " (subscripts must be letters)", 3)
    end
  end
end

local function normalize_einsum(equation, operand_count)
  local compact = equation:gsub("%s+", "")
  local arrow = compact:find("->", 1, true)
  local input, output = compact, nil
  if arrow then
    input = compact:sub(1, arrow - 1)
    output = compact:sub(arrow + 2)
    if output:find("->", 1, true) then
      error("einsum: equation contains more than one '->'", 3)
    end
  end
  local terms = {}
  for term in (input .. ","):gmatch("(.-),") do
    terms[#terms + 1] = term
  end
  if #terms ~= operand_count then
    error(string.format(
      "einsum: equation describes %d operand(s) but %d tensor(s) were given",
      #terms, operand_count), 3)
  end
  local available = {}
  for i, term in ipairs(terms) do
    check_einsum_subscript(term, "operand " .. i)
    for ch in term:gmatch("[a-zA-Z]") do
      available[ch] = true
    end
  end
  if output then
    check_einsum_subscript(output, "the output")
    local seen = {}
    for ch in output:gmatch("[a-zA-Z]") do
      if seen[ch] then
        error("einsum: output subscript '" .. ch ..
          "' appears more than once", 3)
      end
      seen[ch] = true
      if not available[ch] then
        error("einsum: output subscript '" .. ch ..
          "' does not appear in any input operand", 3)
      end
    end
  end
  return compact
end

function torch.einsum(equation, ...)
  if type(equation) ~= "string" then
    error("torch.einsum expects the equation string as its first argument", 2)
  end
  local count = select("#", ...)
  if count == 0 then
    error("torch.einsum expects at least one tensor operand", 2)
  end
  local operands = { ... }
  for i = 1, count do
    if not Tensor.is_tensor(operands[i]) then
      error("einsum expects tensor operands; operand " .. i .. " is " ..
        type(operands[i]), 3)
    end
    if operands[i]:dtype() == "bool" then
      error("einsum does not support bool operands; cast them to a numeric " ..
        "dtype first", 3)
    end
  end
  local first_dtype = operands[1]:dtype()
  for i = 2, count do
    if operands[i]:dtype() ~= first_dtype then
      error(string.format(
        "einsum expects operands with one dtype; operand 1 is %s but " ..
        "operand %d is %s", first_dtype, i, operands[i]:dtype()), 3)
    end
  end
  local normalized = normalize_einsum(equation, count)
  local array = ffi.new("lt_tensor*[?]", count)
  for i = 1, count do
    array[i - 1] = operands[i].h
  end
  return Tensor._wrap(lib.lt_einsum(array, count, normalized))
end

function torch.addmm(input, mat1, mat2, beta, alpha)
  return input:addmm(mat1, mat2, beta, alpha)
end

function torch.baddbmm(input, batch1, batch2, beta, alpha)
  return input:baddbmm(batch1, batch2, beta, alpha)
end

-- threading ------------------------------------------------------------------

function torch.set_num_threads(count)
  if type(count) ~= "number" or count < 1 then
    error("set_num_threads expects a positive integer", 2)
  end
  lib.lt_set_num_threads(count)
end

function torch.get_num_threads()
  return lib.lt_get_num_threads()
end

-- utilities ------------------------------------------------------------------

function torch.manual_seed(seed)
  lib.lt_manual_seed(seed)
  return seed
end

function torch.set_default_dtype(dtype)
  default_dtype = Tensor.resolve_dtype(dtype)
end

function torch.get_default_dtype()
  return default_dtype
end

function torch.save(t, path)
  if not Tensor.is_tensor(t) then
    error("torch.save expects a tensor", 2)
  end
  return t:save(path)
end

function torch.load(path)
  return Tensor._wrap(lib.lt_load(path))
end

function torch.print(t)
  if not Tensor.is_tensor(t) then
    error("torch.print expects a tensor", 2)
  end
  print(t:format())
end

function torch.dtype_name(dtype)
  return Tensor.dtype_names[Tensor.resolve_dtype(dtype)]
end

-- Lazily attach the higher-level packages.  Lazy loading avoids a require
-- cycle when the user starts with `require("torch.nn")`.
setmetatable(torch, {
  __index = function(self, key)
    if key == "nn" or key == "optim" or key == "utils" then
      local package = require("torch." .. key)
      rawset(self, key, package)
      return package
    end
    return nil
  end,
})

local function vector_values(input, name)
  if not Tensor.is_tensor(input) then input = torch.tensor(input) end
  if input:dim() ~= 1 then error(name .. " expects a 1-D tensor", 3) end
  return input
end

function torch.bincount(input, weights, minlength)
  input = vector_values(input, "bincount")
  local values = input:to(torch.int64):totable()
  local weight_values
  if weights ~= nil then
    weights = vector_values(weights, "bincount weights")
    if weights:numel() ~= #values then
      error("bincount weights length mismatch", 3)
    end
    weight_values = weights:to(torch.float64):totable()
  end
  if minlength ~= nil and (type(minlength) ~= "number" or minlength < 0 or
      minlength % 1 ~= 0) then
    error("bincount minlength must be a non-negative integer", 3)
  end
  local maximum = -1
  for _, value in ipairs(values) do
    if value < 0 then error("bincount input must be non-negative", 3) end
    if value > maximum then maximum = value end
  end
  local length = math.max(maximum + 1, minlength or 0)
  local counts = {}
  for i = 1, length do counts[i] = 0 end
  for i, value in ipairs(values) do
    counts[value + 1] = counts[value + 1] +
      (weight_values and weight_values[i] or 1)
  end
  return torch.tensor(counts, { dtype = weights and torch.float64 or torch.int64 })
end

function torch.histc(input, bins, min, max)
  input = vector_values(input, "histc")
  bins = bins or 100
  if type(bins) ~= "number" or bins < 1 or bins % 1 ~= 0 then
    error("histc bins must be a positive integer", 3)
  end
  local values = input:to(torch.float64):totable()
  if #values == 0 then return torch.zeros(bins, { dtype = torch.int64 }) end
  min = min or math.min(table.unpack(values))
  max = max or math.max(table.unpack(values))
  if min > max then error("histc min must not exceed max", 3) end
  local counts = {}
  for i = 1, bins do counts[i] = 0 end
  if min == max then
    for _, value in ipairs(values) do
      if value == min then counts[1] = counts[1] + 1 end
    end
  else
    local width = (max - min) / bins
    for _, value in ipairs(values) do
      if value >= min and value <= max then
        local index = math.floor((value - min) / width) + 1
        if index > bins then index = bins end
        counts[index] = counts[index] + 1
      end
    end
  end
  return torch.tensor(counts, { dtype = torch.int64 })
end

return torch
