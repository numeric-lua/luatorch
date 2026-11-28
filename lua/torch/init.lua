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

torch.VERSION = "0.34.0"
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

function torch.split(tensor, split_size_or_sections, dim)
  if not Tensor.is_tensor(tensor) then
    error("torch.split expects a tensor", 2)
  end
  return tensor:split(split_size_or_sections, dim)
end

function torch.chunk(tensor, chunks, dim)
  if not Tensor.is_tensor(tensor) then
    error("torch.chunk expects a tensor", 2)
  end
  return tensor:chunk(chunks, dim)
end

function torch.unbind(tensor, dim)
  if not Tensor.is_tensor(tensor) then
    error("torch.unbind expects a tensor", 2)
  end
  return tensor:unbind(dim)
end

local function require_tensor_list(tensors, name)
  if type(tensors) ~= "table" or #tensors == 0 then
    error(name .. " expects a non-empty table of tensors", 2)
  end
  for i = 1, #tensors do
    if not Tensor.is_tensor(tensors[i]) then
      error(name .. " expects tensors; entry " .. i .. " is " ..
        type(tensors[i]), 2)
    end
  end
end

function torch.hstack(tensors)
  require_tensor_list(tensors, "hstack")
  local all_1d = true
  for i = 1, #tensors do
    if tensors[i]:dim() ~= 1 then all_1d = false end
  end
  if all_1d then
    return torch.cat(tensors, 1)
  end
  return torch.cat(tensors, 2)
end

function torch.vstack(tensors)
  require_tensor_list(tensors, "vstack")
  local parts = {}
  for i = 1, #tensors do parts[i] = torch.atleast_2d(tensors[i]) end
  return torch.cat(parts, 1)
end

function torch.dstack(tensors)
  require_tensor_list(tensors, "dstack")
  local parts = {}
  for i = 1, #tensors do parts[i] = torch.atleast_3d(tensors[i]) end
  return torch.cat(parts, 3)
end

function torch.column_stack(tensors)
  require_tensor_list(tensors, "column_stack")
  local all_1d = true
  for i = 1, #tensors do
    if tensors[i]:dim() ~= 1 then all_1d = false end
  end
  if all_1d then
    return torch.stack(tensors, 2)
  end
  return torch.cat(tensors, 2)
end

torch.row_stack = torch.vstack

function torch.cartesian_prod(...)
  local tensors = { ... }
  if #tensors == 0 then
    error("cartesian_prod expects at least one tensor", 2)
  end
  local sizes, after = {}, {}
  for i = 1, #tensors do
    local tensor = tensors[i]
    if not Tensor.is_tensor(tensor) or tensor:dim() ~= 1 then
      error("cartesian_prod expects 1-D tensors", 2)
    end
    sizes[i] = tensor:size(1)
  end
  local running = 1
  for i = #tensors, 1, -1 do
    after[i] = running
    running = running * sizes[i]
  end
  local before = 1
  local columns = {}
  for i = 1, #tensors do
    local column = tensors[i]
    if after[i] ~= 1 then
      column = column:repeat_interleave(after[i])
    end
    if before ~= 1 then
      column = column:tile({ before })
    end
    columns[i] = column
    before = before * sizes[i]
  end
  return torch.stack(columns, 2)
end

local function searchsorted_values(sorted, values, right)
  if not Tensor.is_tensor(sorted) or not Tensor.is_tensor(values) then
    error("searchsorted expects two tensors", 2)
  end
  if sorted:dim() ~= 1 then
    error("searchsorted sorted_sequence must be 1-D", 2)
  end
  if right ~= nil and type(right) ~= "boolean" then
    error("searchsorted right must be a boolean", 2)
  end
  local sequence = sorted:totable()
  local flat = values:reshape(-1):totable()
  local result = {}
  for i = 1, #flat do
    local value = flat[i]
    local low, high = 1, #sequence + 1
    while low < high do
      local middle = math.floor((low + high) / 2)
      local move_right
      if right then
        move_right = sequence[middle] <= value
      else
        move_right = sequence[middle] < value
      end
      if move_right then
        low = middle + 1
      else
        high = middle
      end
    end
    result[i] = low
  end
  return Tensor.from_table(result, { dtype = 2 }):reshape(values:sizes())
end

function torch.searchsorted(sorted_sequence, values, right)
  return searchsorted_values(sorted_sequence, values, right)
end

function torch.bucketize(input, boundaries, right)
  if not Tensor.is_tensor(boundaries) or boundaries:dim() ~= 1 then
    error("bucketize boundaries must be a 1-D tensor", 2)
  end
  return searchsorted_values(boundaries, input, right)
end

function torch.where(condition, a, b)
  if not (Tensor.is_tensor(condition) and Tensor.is_tensor(a) and Tensor.is_tensor(b)) then
    error("torch.where expects three tensors", 2)
  end
  return Tensor._wrap(lib.lt_where(condition.h, a.h, b.h))
end

-- Broadcast shape utilities. Shapes are Lua tables; a single table of shape
-- tables is accepted as well as varargs.
function torch.broadcast_shapes(...)
  local shapes = { ... }
  if #shapes == 1 and type(shapes[1]) == "table" and
      (#shapes[1] == 0 or type(shapes[1][1]) == "table") then
    shapes = shapes[1]
  end
  local result = {}
  for i = 1, #shapes do
    local shape = shapes[i]
    if type(shape) ~= "table" then
      error("broadcast_shapes expects shape tables", 2)
    end
    for d = 1, #shape do
      local value = shape[d]
      if type(value) ~= "number" or value ~= value or value % 1 ~= 0 or
          value < 0 or value == math.huge then
        error("broadcast_shapes expects non-negative integer sizes", 2)
      end
    end
    local max_length = math.max(#result, #shape)
    local merged = {}
    for position = 1, max_length do
      local result_index = position - (max_length - #result)
      local shape_index = position - (max_length - #shape)
      local a = result_index >= 1 and result[result_index] or nil
      local b = shape_index >= 1 and shape[shape_index] or nil
      if a == nil then
        merged[position] = b
      elseif b == nil then
        merged[position] = a
      elseif a == b then
        merged[position] = a
      elseif a == 1 then
        merged[position] = b
      elseif b == 1 then
        merged[position] = a
      else
        error("broadcast_shapes: incompatible shapes", 2)
      end
    end
    result = merged
  end
  return result
end

-- Broadcast a list of tensors to a common shape and return expanded views.
function torch.broadcast_tensors(...)
  local tensors = { ... }
  if #tensors == 1 and type(tensors[1]) == "table" and
      (#tensors[1] == 0 or Tensor.is_tensor(tensors[1][1])) then
    tensors = tensors[1]
  end
  if #tensors == 0 then
    error("broadcast_tensors expects at least one tensor", 2)
  end
  local shapes = {}
  for i = 1, #tensors do
    if not Tensor.is_tensor(tensors[i]) then
      error("broadcast_tensors expects tensors", 2)
    end
    shapes[i] = tensors[i]:sizes()
  end
  local shape = torch.broadcast_shapes(unpack(shapes))
  local result = {}
  for i = 1, #tensors do
    local tensor = tensors[i]
    local sizes = {}
    local offset = #shape - tensor:dim()
    for d = 1, #shape do
      local source = d - offset
      if source >= 1 and tensor:size(source) ~= 1 then
        sizes[d] = tensor:size(source)
      else
        sizes[d] = shape[d]
      end
    end
    result[i] = tensor:expand(unpack(sizes))
  end
  return unpack(result)
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
function torch.fmod(a, b)
  if not Tensor.is_tensor(a) then
    error("torch.fmod expects a tensor as its first argument", 2)
  end
  return a:fmod(b)
end
function torch.atan2(a, b)
  if not Tensor.is_tensor(a) then
    error("torch.atan2 expects a tensor as its first argument", 2)
  end
  return a:atan2(b)
end
function torch.copysign(a, b)
  if not Tensor.is_tensor(a) then
    error("torch.copysign expects a tensor as its first argument", 2)
  end
  return a:copysign(b)
end
function torch.floor_divide(a, b)
  if not Tensor.is_tensor(a) then
    error("torch.floor_divide expects a tensor as its first argument", 2)
  end
  return a:floor_divide(b)
end
function torch.heaviside(input, values)
  if not Tensor.is_tensor(input) then
    error("torch.heaviside expects a tensor as its first argument", 2)
  end
  return input:heaviside(values)
end
function torch.addcmul(input, first, second, value)
  if not Tensor.is_tensor(input) then
    error("torch.addcmul expects a tensor as its first argument", 2)
  end
  return input:addcmul(first, second, value)
end
function torch.addcdiv(input, first, second, value)
  if not Tensor.is_tensor(input) then
    error("torch.addcdiv expects a tensor as its first argument", 2)
  end
  return input:addcdiv(first, second, value)
end
function torch.lerp(input, end_tensor, weight)
  if not Tensor.is_tensor(input) then
    error("torch.lerp expects a tensor as its first argument", 2)
  end
  return input:lerp(end_tensor, weight)
end
function torch.logaddexp(input, other)
  if not Tensor.is_tensor(input) then
    error("torch.logaddexp expects a tensor as its first argument", 2)
  end
  return input:logaddexp(other)
end
function torch.logaddexp2(input, other)
  if not Tensor.is_tensor(input) then
    error("torch.logaddexp2 expects a tensor as its first argument", 2)
  end
  return input:logaddexp2(other)
end
function torch.hypot(input, other)
  if not Tensor.is_tensor(input) then
    error("torch.hypot expects a tensor as its first argument", 2)
  end
  return input:hypot(other)
end
function torch.frac(input)
  if not Tensor.is_tensor(input) then
    error("torch.frac expects a tensor", 2)
  end
  return input:frac()
end
torch.matmul = function(a, b) return a:matmul(b) end
torch.mm = function(a, b) return a:mm(b) end
torch.bmm = function(a, b) return a:bmm(b) end
torch.dot = function(a, b) return a:dot(b) end
torch.sum = function(t, dim, keepdim) return t:sum(dim, keepdim) end
torch.mean = function(t, dim, keepdim) return t:mean(dim, keepdim) end
torch.prod = function(t, dim, keepdim) return t:prod(dim, keepdim) end
torch.max = function(t, dim, keepdim) return t:max(dim, keepdim) end
torch.min = function(t, dim, keepdim) return t:min(dim, keepdim) end
function torch.amin(t, dim, keepdim)
  if not Tensor.is_tensor(t) then
    error("torch.amin expects a tensor", 2)
  end
  return t:amin(dim, keepdim)
end
function torch.amax(t, dim, keepdim)
  if not Tensor.is_tensor(t) then
    error("torch.amax expects a tensor", 2)
  end
  return t:amax(dim, keepdim)
end
torch.argmax = function(t, dim, keepdim) return t:argmax(dim, keepdim) end
torch.argmin = function(t, dim, keepdim) return t:argmin(dim, keepdim) end
torch.transpose = function(t, d1, d2) return t:transpose(d1, d2) end
function torch.swapaxes(input, dim0, dim1)
  if not Tensor.is_tensor(input) then
    error("torch.swapaxes expects a tensor", 2)
  end
  return input:swapaxes(dim0, dim1)
end
function torch.moveaxis(input, source, destination)
  if not Tensor.is_tensor(input) then
    error("torch.moveaxis expects a tensor", 2)
  end
  return input:moveaxis(source, destination)
end
torch.movedim = torch.moveaxis

local function atleast_rank(rank, tensors)
  local result = {}
  for i = 1, #tensors do
    local tensor = tensors[i]
    if not Tensor.is_tensor(tensor) then
      error("atleast_* expects tensors", 2)
    end
    if tensor:dim() >= rank then
      result[i] = tensor
    else
      local shape = {}
      if rank == 1 then
        shape = { 1 }
      elseif rank == 2 then
        if tensor:dim() == 0 then
          shape = { 1, 1 }
        else
          shape = { 1, tensor:size(1) }
        end
      else
        if tensor:dim() == 0 then
          shape = { 1, 1, 1 }
        elseif tensor:dim() == 1 then
          shape = { 1, tensor:size(1), 1 }
        else
          shape = { tensor:size(1), tensor:size(2), 1 }
        end
      end
      result[i] = tensor:reshape(shape)
    end
  end
  return unpack(result)
end

function torch.atleast_1d(...) return atleast_rank(1, { ... }) end
function torch.atleast_2d(...) return atleast_rank(2, { ... }) end
function torch.atleast_3d(...) return atleast_rank(3, { ... }) end
torch.reshape = function(t, ...) return t:reshape(...) end
torch.clamp = function(t, lo, hi) return t:clamp(lo, hi) end
function torch.clamp_min(input, minimum)
  if not Tensor.is_tensor(input) then
    error("torch.clamp_min expects a tensor", 2)
  end
  return input:clamp_min(minimum)
end
function torch.clamp_max(input, maximum)
  if not Tensor.is_tensor(input) then
    error("torch.clamp_max expects a tensor", 2)
  end
  return input:clamp_max(maximum)
end
function torch.logit(input, eps)
  if not Tensor.is_tensor(input) then
    error("torch.logit expects a tensor", 2)
  end
  return input:logit(eps)
end
torch.relu = function(t) return t:relu() end
torch.sigmoid = function(t) return t:sigmoid() end
torch.tanh = function(t) return t:tanh() end
function torch.softplus(input, beta, threshold)
  if not Tensor.is_tensor(input) then
    error("torch.softplus expects a tensor", 2)
  end
  return input:softplus(beta, threshold)
end
function torch.elu(input, alpha)
  if not Tensor.is_tensor(input) then
    error("torch.elu expects a tensor", 2)
  end
  return input:elu(alpha)
end
function torch.selu(input)
  if not Tensor.is_tensor(input) then
    error("torch.selu expects a tensor", 2)
  end
  return input:selu()
end
function torch.silu(input)
  if not Tensor.is_tensor(input) then
    error("torch.silu expects a tensor", 2)
  end
  return input:silu()
end
function torch.gelu(input, approximate)
  if not Tensor.is_tensor(input) then
    error("torch.gelu expects a tensor", 2)
  end
  return input:gelu(approximate)
end
function torch.deg2rad(input)
  if not Tensor.is_tensor(input) then
    error("torch.deg2rad expects a tensor", 2)
  end
  return input:deg2rad()
end
function torch.rad2deg(input)
  if not Tensor.is_tensor(input) then
    error("torch.rad2deg expects a tensor", 2)
  end
  return input:rad2deg()
end
function torch.softsign(input)
  if not Tensor.is_tensor(input) then
    error("torch.softsign expects a tensor", 2)
  end
  return input:softsign()
end
function torch.hardsigmoid(input)
  if not Tensor.is_tensor(input) then
    error("torch.hardsigmoid expects a tensor", 2)
  end
  return input:hardsigmoid()
end
function torch.hardswish(input)
  if not Tensor.is_tensor(input) then
    error("torch.hardswish expects a tensor", 2)
  end
  return input:hardswish()
end
torch.exp = function(t) return t:exp() end
torch.expm1 = function(t) return t:expm1() end
torch.log = function(t) return t:log() end
torch.log2 = function(t) return t:log2() end
torch.log10 = function(t) return t:log10() end
torch.erf = function(t) return t:erf() end
torch.erfc = function(t) return t:erfc() end
torch.erfinv = function(t) return t:erfinv() end
torch.i0 = function(t) return t:i0() end
torch.i0e = function(t) return t:i0e() end
torch.i1 = function(t) return t:i1() end
torch.i1e = function(t) return t:i1e() end
torch.lgamma = function(t) return t:lgamma() end
torch.digamma = function(t) return t:digamma() end
torch.ndtr = function(t) return t:ndtr() end
torch.log_ndtr = function(t) return t:log_ndtr() end
torch.ndtri = function(t) return t:ndtri() end
torch.sinc = function(t) return t:sinc() end
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

function torch.allclose(a, b, rtol, atol, equal_nan)
  return a:allclose(b, rtol, atol, equal_nan)
end

function torch.isclose(a, b, rtol, atol, equal_nan)
  if not Tensor.is_tensor(a) then
    error("torch.isclose expects a tensor as its first argument", 2)
  end
  return a:isclose(b, rtol, atol, equal_nan)
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

function torch.isposinf(t)
  if not Tensor.is_tensor(t) then
    error("torch.isposinf expects a tensor", 2)
  end
  return t:isposinf()
end

function torch.isneginf(t)
  if not Tensor.is_tensor(t) then
    error("torch.isneginf expects a tensor", 2)
  end
  return t:isneginf()
end

function torch.nan_to_num(t, nan, posinf, neginf)
  if not Tensor.is_tensor(t) then
    error("torch.nan_to_num expects a tensor", 2)
  end
  return t:nan_to_num(nan, posinf, neginf)
end

-- Reductions -----------------------------------------------------------------

function torch.aminmax(t, dim, keepdim)
  if not Tensor.is_tensor(t) then
    error("torch.aminmax expects a tensor", 2)
  end
  return t:aminmax(dim, keepdim)
end

function torch.count_nonzero(t, dim)
  if not Tensor.is_tensor(t) then
    error("torch.count_nonzero expects a tensor", 2)
  end
  return t:count_nonzero(dim)
end

function torch.nansum(t, dim, keepdim)
  if not Tensor.is_tensor(t) then
    error("torch.nansum expects a tensor", 2)
  end
  return t:nansum(dim, keepdim)
end

function torch.nanmean(t, dim, keepdim)
  if not Tensor.is_tensor(t) then
    error("torch.nanmean expects a tensor", 2)
  end
  return t:nanmean(dim, keepdim)
end

function torch.nanvar(t, dim, unbiased, keepdim)
  if not Tensor.is_tensor(t) then
    error("torch.nanvar expects a tensor", 2)
  end
  return t:nanvar(dim, unbiased, keepdim)
end

function torch.nanstd(t, dim, unbiased, keepdim)
  if not Tensor.is_tensor(t) then
    error("torch.nanstd expects a tensor", 2)
  end
  return t:nanstd(dim, unbiased, keepdim)
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

function torch.argsort(tensor, dim, descending)
  if not Tensor.is_tensor(tensor) then
    error("torch.argsort expects a tensor", 2)
  end
  return tensor:argsort(dim, descending)
end

function torch.diff(tensor, n, dim)
  if not Tensor.is_tensor(tensor) then
    error("torch.diff expects a tensor", 2)
  end
  return tensor:diff(n, dim)
end

function torch.roll(tensor, shifts, dims)
  if not Tensor.is_tensor(tensor) then
    error("torch.roll expects a tensor", 2)
  end
  return tensor:roll(shifts, dims)
end

function torch.tile(tensor, reps)
  if not Tensor.is_tensor(tensor) then
    error("torch.tile expects a tensor", 2)
  end
  return tensor:tile(reps)
end

function torch.repeat_interleave(tensor, repeats, dim)
  if not Tensor.is_tensor(tensor) then
    error("torch.repeat_interleave expects a tensor", 2)
  end
  return tensor:repeat_interleave(repeats, dim)
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

function torch.take(tensor, index)
  if not Tensor.is_tensor(tensor) then
    error("torch.take expects a tensor as its first argument", 2)
  end
  return tensor:take(index)
end

function torch.scatter(tensor, dim, index, source)
  return tensor:scatter(dim, index, source)
end

function torch.unique(tensor, sorted_flag, return_inverse, return_counts)
  return tensor:unique(sorted_flag, return_inverse, return_counts)
end

function torch.unique_consecutive(tensor, return_inverse, return_counts)
  if not Tensor.is_tensor(tensor) then
    error("torch.unique_consecutive expects a tensor", 2)
  end
  return tensor:unique_consecutive(return_inverse, return_counts)
end

function torch.fmax(a, b)
  if not Tensor.is_tensor(a) then
    error("torch.fmax expects a tensor as its first argument", 2)
  end
  return a:fmax(b)
end

function torch.fmin(a, b)
  if not Tensor.is_tensor(a) then
    error("torch.fmin expects a tensor as its first argument", 2)
  end
  return a:fmin(b)
end

function torch.kthvalue(tensor, k, dim, keepdim)
  if not Tensor.is_tensor(tensor) then
    error("torch.kthvalue expects a tensor", 2)
  end
  return tensor:kthvalue(k, dim, keepdim)
end

function torch.mode(tensor, dim, keepdim)
  if not Tensor.is_tensor(tensor) then
    error("torch.mode expects a tensor", 2)
  end
  return tensor:mode(dim, keepdim)
end

function torch.nanmedian(tensor, dim, keepdim)
  if not Tensor.is_tensor(tensor) then
    error("torch.nanmedian expects a tensor", 2)
  end
  return tensor:nanmedian(dim, keepdim)
end

function torch.nanquantile(tensor, q, dim, keepdim)
  if not Tensor.is_tensor(tensor) then
    error("torch.nanquantile expects a tensor", 2)
  end
  return tensor:nanquantile(q, dim, keepdim)
end

function torch.logcumsumexp(tensor, dim)
  if not Tensor.is_tensor(tensor) then
    error("torch.logcumsumexp expects a tensor", 2)
  end
  return tensor:logcumsumexp(dim)
end

function torch.cummax(tensor, dim)
  if not Tensor.is_tensor(tensor) then
    error("torch.cummax expects a tensor", 2)
  end
  return tensor:cummax(dim)
end

function torch.cummin(tensor, dim)
  if not Tensor.is_tensor(tensor) then
    error("torch.cummin expects a tensor", 2)
  end
  return tensor:cummin(dim)
end

function torch.entr(tensor)
  if not Tensor.is_tensor(tensor) then
    error("torch.entr expects a tensor", 2)
  end
  return tensor:entr()
end

function torch.xlogy(a, b)
  if not Tensor.is_tensor(a) then
    error("torch.xlogy expects a tensor as its first argument", 2)
  end
  return a:xlogy(b)
end

function torch.xlog1py(a, b)
  if not Tensor.is_tensor(a) then
    error("torch.xlog1py expects a tensor as its first argument", 2)
  end
  return a:xlog1py(b)
end

function torch.multinomial(tensor, num_samples, replacement)
  return tensor:multinomial(num_samples, replacement)
end

-- linear algebra -------------------------------------------------------------

-- Resolve a 1-based linalg dimension, accepting negative values. Kept local
-- so matrix_norm and the norm dispatcher validate dimensions identically.
local function linalg_dim(value, rank, name)
  if type(value) ~= "number" or value ~= value or value % 1 ~= 0 or
      value == math.huge or value == -math.huge then
    error(name .. " dimension must be an integer", 3)
  end
  local dim = math.floor(value)
  if dim < 0 then dim = rank + dim + 1 end
  if dim < 1 or dim > rank then
    error(string.format("%s dimension out of range (got %s, expected 1..%d)",
      name, tostring(value), rank), 3)
  end
  return dim
end

-- Compute a matrix norm over an arbitrary pair of dimensions. The selected
-- dimensions are moved to the end, batch dimensions are flattened only for
-- the SVD-backed orders, and keepdim restores the original rank with the two
-- reduced dimensions set to one.
local function matrix_norm_tensor(a, ord, dim1, dim2, keepdim)
  if ord == nil then ord = "fro" end
  local rank = a:dim()
  if rank < 2 then
    error("matrix_norm expects a tensor with at least 2 dimensions", 3)
  end
  if dim1 == nil then dim1 = rank - 1 end
  if dim2 == nil then dim2 = rank end
  dim1 = linalg_dim(dim1, rank, "matrix_norm")
  dim2 = linalg_dim(dim2, rank, "matrix_norm")
  if dim1 == dim2 then
    error("matrix_norm dims must be different", 3)
  end

  local valid_ord = ord == "fro" or ord == "nuc" or ord == 1 or ord == -1 or
    ord == 2 or ord == -2 or ord == math.huge or ord == -math.huge
  if not valid_ord then
    error("matrix_norm supports 'fro', 'nuc', 1, -1, 2, -2, inf, and -inf", 3)
  end
  if ord == "nuc" or ord == 2 or ord == -2 then
    local dtype = a:dtype()
    if dtype ~= "float32" and dtype ~= "float64" then
      error("matrix_norm spectral and nuclear orders support float32 and " ..
        "float64 tensors", 3)
    end
  end

  local permutation = {}
  for i = 1, rank do
    if i ~= dim1 and i ~= dim2 then
      permutation[#permutation + 1] = i
    end
  end
  permutation[#permutation + 1] = dim1
  permutation[#permutation + 1] = dim2
  local permuted = a:permute(unpack(permutation))

  local batch_shape = {}
  for i = 1, rank - 2 do batch_shape[i] = permuted:size(i) end
  local rows, columns = permuted:size(-2), permuted:size(-1)

  local result
  if ord == "fro" then
    result = permuted:pow(2):sum(-1):sum(-1):sqrt()
  elseif ord == 1 or ord == -1 then
    local sums = permuted:abs():sum(-2)
    result = ord == 1 and sums:max(-1) or sums:min(-1)
  elseif ord == math.huge or ord == -math.huge then
    local sums = permuted:abs():sum(-1)
    result = ord == math.huge and sums:max(-1) or sums:min(-1)
  else
    local count = 1
    for i = 1, #batch_shape do count = count * batch_shape[i] end
    if count == 0 then
      -- An empty batch has no matrices to decompose; preserve its shape.
      result = permuted:pow(2):sum(-1):sum(-1):sqrt()
    else
      local _, singular = permuted:reshape(count, rows, columns):svd(false)
      if ord == "nuc" then
        result = singular:sum(-1)
      elseif ord == 2 then
        result = singular:max(-1)
      else
        result = singular:min(-1)
      end
    end
  end

  if result:dim() ~= #batch_shape then
    result = result:reshape(batch_shape)
  end
  if keepdim then
    local shape = {}
    for i = 1, rank do
      if i == dim1 or i == dim2 then
        shape[i] = 1
      else
        shape[i] = a:size(i)
      end
    end
    result = result:reshape(shape)
  end
  return result
end

-- Vector norm with the full real-valued order set. Positive orders and +inf
-- reuse the native reduction; zero counts nonzeros; negative finite orders are
-- built from abs/pow/sum, and -inf is the minimum absolute value. A missing
-- dim reduces every dimension, matching the flattening behavior of norm.
local function vector_norm_tensor(a, ord, dim, keepdim)
  ord = ord == nil and 2 or ord
  if type(ord) ~= "number" then
    error("vector_norm ord must be a number", 3)
  end
  local rank = a:dim()
  local resolved = dim
  if resolved ~= nil then
    resolved = linalg_dim(resolved, rank, "vector_norm")
  end

  local result
  if ord == 0 then
    result = a:ne(0):to(a:dtype_id()):sum(resolved, keepdim)
  elseif ord == math.huge then
    result = a:abs():max(resolved, keepdim)
  elseif ord == -math.huge then
    result = a:abs():min(resolved, keepdim)
  elseif ord > 0 then
    result = a:norm(ord, resolved, keepdim)
  else
    result = a:abs():pow(ord):sum(resolved, keepdim):pow(1 / ord)
  end
  if keepdim and dim == nil then
    local shape = {}
    for i = 1, rank do shape[i] = 1 end
    result = result:reshape(shape)
  end
  return result
end

-- Count singular values (or absolute eigenvalues for hermitian inputs) above
-- a threshold. The default threshold follows PyTorch's rank convention:
-- max(m, n) * eps(dtype) * largest value. A supplied tol is used as an
-- absolute cutoff and a supplied rtol as rtol * largest.
local function matrix_rank_values(values, tol, rows, columns, dtype, rtol)
  local largest = 0
  for i = 1, #values do
    if values[i] > largest then largest = values[i] end
  end
  local threshold
  if rtol ~= nil then
    threshold = rtol * largest
  elseif tol ~= nil then
    threshold = tol
  else
    local epsilon = dtype == "float32" and 1.1920929e-7 or
      2.220446049250313e-16
    threshold = math.max(rows, columns) * epsilon * largest
  end
  local rank = 0
  for i = 1, #values do
    if values[i] > threshold then rank = rank + 1 end
  end
  return rank
end

-- Rank of a matrix or batch of matrices from its singular values. A 2-D input
-- keeps the historical Lua-number result; batches return an int64 tensor with
-- the batch shape. hermitian=true uses absolute eigenvalues of a real
-- symmetric square matrix instead of the SVD.
local function matrix_rank_tensor(a, tol, hermitian, rtol)
  if not Tensor.is_tensor(a) then
    error("matrix_rank expects a tensor", 3)
  end
  if a:dim() < 2 then
    error("matrix_rank expects a matrix or a batch of matrices", 3)
  end
  local dtype = a:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("matrix_rank supports float32 and float64 tensors", 3)
  end
  if hermitian ~= nil and type(hermitian) ~= "boolean" then
    error("matrix_rank hermitian must be a boolean", 3)
  end
  if tol ~= nil and (type(tol) ~= "number" or tol ~= tol or tol < 0 or
      tol == math.huge) then
    error("matrix_rank tol must be a non-negative finite number", 3)
  end
  if rtol ~= nil and (type(rtol) ~= "number" or rtol ~= rtol or rtol < 0 or
      rtol == math.huge) then
    error("matrix_rank rtol must be a non-negative finite number", 3)
  end
  if tol ~= nil and rtol ~= nil then
    error("matrix_rank accepts either tol or rtol, not both", 3)
  end
  local rows, columns = a:size(-2), a:size(-1)
  if hermitian and rows ~= columns then
    error("matrix_rank hermitian=true expects square matrices", 3)
  end

  local values
  if hermitian then
    values = a:eigvalsh("L"):abs()
  else
    values = a:svdvals()
  end

  if a:dim() == 2 then
    return matrix_rank_values(values:totable(), tol, rows, columns, dtype,
      rtol)
  end

  local batch_shape = {}
  for i = 1, a:dim() - 2 do batch_shape[i] = a:size(i) end
  local batch_count = 1
  for i = 1, #batch_shape do
    batch_count = batch_count * batch_shape[i]
  end
  if batch_count == 0 then
    return torch.zeros(batch_shape, { dtype = torch.int64 })
  end

  local k = math.min(rows, columns)
  local flat = values:reshape(batch_count, k)
  local ranks = {}
  for i = 1, batch_count do
    ranks[i] = matrix_rank_values(flat:select(1, i):totable(), tol,
      rows, columns, dtype, rtol)
  end
  return torch.tensor(ranks, { dtype = torch.int64 }):reshape(batch_shape)
end

-- Condition number of a matrix or batch of matrices. The spectral orders
-- (2 and -2) use singular values and accept rectangular inputs; every other
-- order multiplies the norm of the input by the norm of its inverse, so it
-- requires square matrices. -2 is smallest/largest singular value, matching
-- PyTorch's reciprocal condition number for that order.
local function cond_tensor(a, p)
  if not Tensor.is_tensor(a) then
    error("cond expects a tensor", 3)
  end
  if a:dim() < 2 then
    error("cond expects a matrix or a batch of matrices", 3)
  end
  local dtype = a:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("cond supports float32 and float64 tensors", 3)
  end
  if p == nil then p = 2 end
  if type(p) == "string" then p = p:lower() end
  local square = a:size(-2) == a:size(-1)

  if p == 2 then
    return matrix_norm_tensor(a, 2, -2, -1) /
      matrix_norm_tensor(a, -2, -2, -1)
  elseif p == -2 then
    return matrix_norm_tensor(a, -2, -2, -1) /
      matrix_norm_tensor(a, 2, -2, -1)
  end

  local valid = p == 1 or p == -1 or p == math.huge or p == -math.huge or
    p == "fro" or p == "nuc"
  if not valid then
    error("cond supports 2, -2, 1, -1, inf, -inf, 'fro', and 'nuc'", 3)
  end
  if not square then
    error("cond with p=" .. tostring(p) .. " expects square matrices", 3)
  end
  local inverse = a:inverse()
  return matrix_norm_tensor(a, p, -2, -1) *
    matrix_norm_tensor(inverse, p, -2, -1)
end

torch.linalg = {
  matmul = function(a, b)
    if not Tensor.is_tensor(a) or not Tensor.is_tensor(b) then
      error("matmul expects two tensors", 3)
    end
    return a:matmul(b)
  end,
  mm = function(a, b)
    if not Tensor.is_tensor(a) or not Tensor.is_tensor(b) then
      error("mm expects two tensors", 3)
    end
    return a:mm(b)
  end,
  mv = function(a, b)
    if not Tensor.is_tensor(a) or not Tensor.is_tensor(b) then
      error("mv expects two tensors", 3)
    end
    return a:mv(b)
  end,
  outer = function(a, b)
    if not Tensor.is_tensor(a) or not Tensor.is_tensor(b) then
      error("outer expects two tensors", 3)
    end
    return a:outer(b)
  end,
  inv = function(a) return a:inverse() end,
  inv_ex = function(a, opts)
    if not Tensor.is_tensor(a) then
      error("inv_ex expects a tensor", 3)
    end
    return a:inv_ex(opts)
  end,
  tensorinv = function(a, ind) return a:tensorinv(ind) end,
  tensorsolve = function(a, b, dims) return a:tensorsolve(b, dims) end,
  det = function(a) return a:det() end,
  slogdet = function(a) return a:slogdet() end,
  solve = function(a, b, opts) return a:solve(b, opts) end,
  solve_ex = function(a, b, opts)
    if not Tensor.is_tensor(a) then
      error("solve_ex expects a tensor", 3)
    end
    return a:solve_ex(b, opts)
  end,
  solve_triangular = function(a, b, opts)
    return a:solve_triangular(b, opts)
  end,
  lstsq = function(a, b, opts) return a:lstsq(b, opts) end,
  cholesky = function(a, upper) return a:cholesky(upper) end,
  cholesky_inverse = function(a, upper)
    if not Tensor.is_tensor(a) then
      error("cholesky_inverse expects a tensor", 3)
    end
    return a:cholesky_inverse(upper)
  end,
  cholesky_ex = function(a, opts)
    if not Tensor.is_tensor(a) then
      error("cholesky_ex expects a tensor", 3)
    end
    return a:cholesky_ex(opts)
  end,
  eigh = function(a, uplo)
    if not Tensor.is_tensor(a) then
      error("eigh expects a tensor", 3)
    end
    return a:eigh(uplo)
  end,
  eigvalsh = function(a, uplo)
    if not Tensor.is_tensor(a) then
      error("eigvalsh expects a tensor", 3)
    end
    return a:eigvalsh(uplo)
  end,
  svd = function(a, full_matrices) return a:svd(full_matrices) end,
  svdvals = function(a) return a:svdvals() end,
  qr = function(a, mode) return a:qr(mode) end,
  householder_product = function(a, tau)
    return a:householder_product(tau)
  end,
  lu = function(a) return a:lu() end,
  lu_factor = function(a) return a:lu_factor() end,
  lu_factor_ex = function(a, opts)
    if not Tensor.is_tensor(a) then
      error("lu_factor_ex expects a tensor", 3)
    end
    return a:lu_factor_ex(opts)
  end,
  lu_solve = function(lu, pivots, b, opts)
    return lu:lu_solve(pivots, b, opts)
  end,
  lu_unpack = function(lu, pivots, opts)
    if not Tensor.is_tensor(lu) then
      error("lu_unpack expects a packed LU tensor", 3)
    end
    return lu:lu_unpack(pivots, opts)
  end,
  ldl_factor = function(a, opts)
    if not Tensor.is_tensor(a) then
      error("ldl_factor expects a tensor", 3)
    end
    return a:ldl_factor(opts)
  end,
  ldl_factor_ex = function(a, opts)
    if not Tensor.is_tensor(a) then
      error("ldl_factor_ex expects a tensor", 3)
    end
    return a:ldl_factor_ex(opts)
  end,
  ldl_solve = function(ld, pivots, b, opts)
    return ld:ldl_solve(pivots, b, opts)
  end,
  pinv = function(a, opts) return a:pinv(opts) end,
  matrix_exp = function(a) return a:matrix_exp() end,
  matrix_power = function(a, n) return a:matrix_power(n) end,
  diagonal = function(a, offset, dim1, dim2)
    if not Tensor.is_tensor(a) then
      error("diagonal expects a tensor", 3)
    end
    return a:diagonal(offset, dim1 == nil and -2 or dim1,
      dim2 == nil and -1 or dim2)
  end,
  vander = function(x, N)
    if not Tensor.is_tensor(x) then
      error("vander expects a tensor", 3)
    end
    return x:vander(N, true)
  end,
  cross = function(a, b, dim)
    if not Tensor.is_tensor(a) or not Tensor.is_tensor(b) then
      error("cross expects two tensors", 3)
    end
    return a:cross(b, dim)
  end,
  vecdot = function(x, y, dim)
    if not Tensor.is_tensor(x) or not Tensor.is_tensor(y) then
      error("vecdot expects two tensors", 3)
    end
    return x:vecdot(y, dim)
  end,
  vector_norm = function(a, ord, dim, keepdim)
    if not Tensor.is_tensor(a) then
      error("vector_norm expects a tensor", 3)
    end
    return vector_norm_tensor(a, ord, dim, keepdim)
  end,
  norm = function(a, ord, dim, keepdim)
    if not Tensor.is_tensor(a) then
      error("norm expects a tensor", 3)
    end
    if dim == nil then
      if ord == nil then
        return vector_norm_tensor(a, 2, nil, keepdim)
      end
      local rank = a:dim()
      if rank == 1 then
        return vector_norm_tensor(a, ord, 1, keepdim)
      elseif rank == 2 then
        return matrix_norm_tensor(a, ord, 1, 2, keepdim)
      end
      error("norm: if dim is not specified but ord is, the input must be " ..
        "1-D or 2-D", 3)
    elseif type(dim) == "number" then
      return vector_norm_tensor(a, ord, dim, keepdim)
    elseif type(dim) == "table" then
      if #dim == 1 then
        return vector_norm_tensor(a, ord, dim[1], keepdim)
      elseif #dim == 2 then
        return matrix_norm_tensor(a, ord, dim[1], dim[2], keepdim)
      end
      error("norm dim must be an integer or a 1- or 2-element table", 3)
    end
    error("norm dim must be an integer or a table of dimensions", 3)
  end,
  cond = function(a, p)
    local result = cond_tensor(a, p)
    -- Preserve the historical Lua-number result for a plain 2-D matrix.
    if result:dim() == 0 then return result:item() end
    return result
  end,
  matrix_rank = function(a, tol, hermitian)
    if type(tol) == "table" then
      if hermitian ~= nil then
        error("matrix_rank options table cannot be combined with the " ..
          "hermitian argument", 3)
      end
      for key in pairs(tol) do
        if key ~= "tol" and key ~= "rtol" and key ~= "hermitian" then
          error("matrix_rank got unknown option '" .. tostring(key) .. "'", 3)
        end
      end
      return matrix_rank_tensor(a, tol.tol, tol.hermitian, tol.rtol)
    end
    return matrix_rank_tensor(a, tol, hermitian)
  end,
  matrix_norm = function(a, ord, dim1, dim2, keepdim)
    if not Tensor.is_tensor(a) then
      error("matrix_norm expects a tensor", 3)
    end
    local default_dims = dim1 == nil and dim2 == nil
    local result = matrix_norm_tensor(a, ord, dim1, dim2, keepdim)
    -- Preserve the historical Lua-number result for a plain 2-D matrix.
    if default_dims and not keepdim and result:dim() == 0 then
      return result:item()
    end
    return result
  end,
}

-- Multiply two or more matrices with an optimal parenthesization. The first
-- and last operands may be 1-D vectors, matching torch.linalg.multi_dot; all
-- middle operands must be 2-D. The dynamic program minimizes scalar
-- multiplications before building the expression, and matmul carries the
-- autograd graph.
function torch.linalg.multi_dot(tensors)
  if type(tensors) ~= "table" then
    error("multi_dot expects a table of tensors", 3)
  end
  local count = #tensors
  if count < 2 then
    error("multi_dot expects at least two tensors", 3)
  end
  for i = 1, count do
    if not Tensor.is_tensor(tensors[i]) then
      error("multi_dot expects tensors; entry " .. i .. " is " ..
        type(tensors[i]), 3)
    end
  end
  local dtype = tensors[1]:dtype()
  if dtype ~= "float16" and dtype ~= "float32" and dtype ~= "float64" then
    error("multi_dot supports float16, float32, and float64 tensors", 3)
  end
  for i = 2, count do
    if tensors[i]:dtype() ~= dtype then
      error("multi_dot expects all tensors to share one dtype", 3)
    end
  end
  for i = 1, count do
    local rank = tensors[i]:dim()
    if i == 1 or i == count then
      if rank ~= 1 and rank ~= 2 then
        error("multi_dot expects the first and last tensors to be 1-D or 2-D", 3)
      end
    elseif rank ~= 2 then
      error("multi_dot expects middle tensors to be 2-D", 3)
    end
  end

  local matrices = {}
  for i = 1, count do matrices[i] = tensors[i] end
  local first_vector = matrices[1]:dim() == 1
  local last_vector = matrices[count]:dim() == 1
  if first_vector then
    matrices[1] = matrices[1]:reshape(1, matrices[1]:size(1))
  end
  if last_vector then
    matrices[count] = matrices[count]:reshape(matrices[count]:size(1), 1)
  end
  for i = 1, count - 1 do
    if matrices[i]:size(2) ~= matrices[i + 1]:size(1) then
      error(string.format(
        "multi_dot: tensors %d and %d with shapes [%d, %d] and [%d, %d] " ..
        "cannot be multiplied", i, i + 1,
        matrices[i]:size(1), matrices[i]:size(2),
        matrices[i + 1]:size(1), matrices[i + 1]:size(2)), 3)
    end
  end

  local dimensions = { matrices[1]:size(1) }
  for i = 1, count do
    dimensions[i + 1] = matrices[i]:size(2)
  end
  local cost, split = {}, {}
  for i = 1, count do
    cost[i], split[i] = {}, {}
    cost[i][i] = 0
  end
  for length = 2, count do
    for i = 1, count - length + 1 do
      local j = i + length - 1
      local best, best_split
      for k = i, j - 1 do
        local candidate = cost[i][k] + cost[k + 1][j] +
          dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
        if best == nil or candidate < best then
          best, best_split = candidate, k
        end
      end
      cost[i][j], split[i][j] = best, best_split
    end
  end

  local function multiply(i, j)
    if i == j then return matrices[i] end
    local k = split[i][j]
    return multiply(i, k):matmul(multiply(k + 1, j))
  end
  local result = multiply(1, count)
  if first_vector then result = result:squeeze(1) end
  if last_vector then result = result:squeeze(-1) end
  return result
end

-- Contract the last `dims` dimensions of `a` with the first `dims` of `b`,
-- or the dimensions named by a {a_dims, b_dims} pair. Free dimensions keep
-- their order in the result: a's free dimensions followed by b's. The
-- contraction is a permute/reshape/matmul, so autograd works throughout.
function torch.tensordot(a, b, dims)
  if not Tensor.is_tensor(a) or not Tensor.is_tensor(b) then
    error("tensordot expects two tensors", 3)
  end
  if a:dim() < 1 or b:dim() < 1 then
    error("tensordot expects tensors with at least one dimension", 3)
  end
  if a:dtype_id() ~= b:dtype_id() then
    error("tensordot expects both tensors to share a dtype", 3)
  end
  if dims == nil then dims = 2 end

  local a_dims, b_dims = {}, {}
  if type(dims) == "number" then
    if dims ~= dims or dims % 1 ~= 0 or dims < 0 or dims == math.huge then
      error("tensordot dims must be a non-negative integer", 3)
    end
    dims = math.floor(dims)
    if dims > a:dim() or dims > b:dim() then
      error("tensordot dims exceeds the rank of an input", 3)
    end
    for i = 1, dims do
      a_dims[i] = a:dim() - dims + i
      b_dims[i] = i
    end
  elseif type(dims) == "table" then
    if #dims ~= 2 or type(dims[1]) ~= "table" or
        type(dims[2]) ~= "table" then
      error("tensordot dims must be an integer or a {a_dims, b_dims} pair", 3)
    end
    if #dims[1] ~= #dims[2] then
      error("tensordot dims lists must have the same length", 3)
    end
    for i = 1, #dims[1] do
      a_dims[i] = linalg_dim(dims[1][i], a:dim(), "tensordot")
      b_dims[i] = linalg_dim(dims[2][i], b:dim(), "tensordot")
    end
  else
    error("tensordot dims must be an integer or a table", 3)
  end

  local a_seen, b_seen = {}, {}
  for i = 1, #a_dims do
    if a_seen[a_dims[i]] then
      error("tensordot a_dims must not repeat dimensions", 3)
    end
    if b_seen[b_dims[i]] then
      error("tensordot b_dims must not repeat dimensions", 3)
    end
    a_seen[a_dims[i]] = true
    b_seen[b_dims[i]] = true
  end
  for i = 1, #a_dims do
    local a_size = a:size(a_dims[i])
    local b_size = b:size(b_dims[i])
    if a_size ~= b_size then
      error(string.format(
        "tensordot dimension %d has mismatched sizes: %d vs %d",
        i, a_size, b_size), 3)
    end
  end

  local a_free, b_free = {}, {}
  for i = 1, a:dim() do
    if not a_seen[i] then a_free[#a_free + 1] = i end
  end
  for i = 1, b:dim() do
    if not b_seen[i] then b_free[#b_free + 1] = i end
  end

  local function size_product(tensor, dimensions)
    local product = 1
    for i = 1, #dimensions do
      product = product * tensor:size(dimensions[i])
    end
    return product
  end
  local rows = size_product(a, a_free)
  local inner = size_product(a, a_dims)
  local columns = size_product(b, b_free)

  local a_permutation, b_permutation = {}, {}
  for i = 1, #a_free do a_permutation[#a_permutation + 1] = a_free[i] end
  for i = 1, #a_dims do a_permutation[#a_permutation + 1] = a_dims[i] end
  for i = 1, #b_dims do b_permutation[#b_permutation + 1] = b_dims[i] end
  for i = 1, #b_free do b_permutation[#b_permutation + 1] = b_free[i] end

  local left = a:permute(unpack(a_permutation)):reshape(rows, inner)
  local right = b:permute(unpack(b_permutation)):reshape(inner, columns)
  local out_shape = {}
  for i = 1, #a_free do
    out_shape[#out_shape + 1] = a:size(a_free[i])
  end
  for i = 1, #b_free do
    out_shape[#out_shape + 1] = b:size(b_free[i])
  end
  return left:matmul(right):reshape(out_shape)
end

function torch.cholesky_inverse(a, upper)
  if not Tensor.is_tensor(a) then
    error("torch.cholesky_inverse expects a tensor", 2)
  end
  return a:cholesky_inverse(upper)
end

function torch.cholesky_solve(b, factor, upper)
  if not Tensor.is_tensor(b) then
    error("torch.cholesky_solve expects a right-hand side tensor", 2)
  end
  return b:cholesky_solve(factor, upper)
end

function torch.lu_unpack(lu, pivots, opts)
  if not Tensor.is_tensor(lu) then
    error("torch.lu_unpack expects a packed LU tensor", 2)
  end
  return lu:lu_unpack(pivots, opts)
end

function torch.diagonal(a, offset, dim1, dim2)
  if not Tensor.is_tensor(a) then
    error("torch.diagonal expects a tensor", 2)
  end
  return a:diagonal(offset, dim1, dim2)
end

function torch.tril(a, diagonal)
  if not Tensor.is_tensor(a) then
    error("torch.tril expects a tensor", 2)
  end
  return a:tril(diagonal)
end

function torch.triu(a, diagonal)
  if not Tensor.is_tensor(a) then
    error("torch.triu expects a tensor", 2)
  end
  return a:triu(diagonal)
end

local function triangular_indices(row, col, diagonal, opts, keep_lower, name)
  local function size_argument(value, which)
    if type(value) ~= "number" or value ~= value or value % 1 ~= 0 or
        value < 0 or value == math.huge then
      error(name .. " " .. which .. " must be a non-negative integer", 3)
    end
    return math.floor(value)
  end
  row = size_argument(row, "row")
  col = size_argument(col, "col")
  diagonal = diagonal == nil and 0 or diagonal
  if type(diagonal) ~= "number" or diagonal ~= diagonal or
      diagonal % 1 ~= 0 or diagonal == math.huge or diagonal == -math.huge then
    error(name .. " diagonal must be an integer", 3)
  end
  diagonal = math.floor(diagonal)

  local dtype
  if opts ~= nil then
    if type(opts) ~= "table" then
      error(name .. " options must be a table", 3)
    end
    for key in pairs(opts) do
      if key ~= "dtype" and key ~= "device" then
        error(name .. " got unknown option '" .. tostring(key) .. "'", 3)
      end
    end
    if opts.device ~= nil and opts.device ~= "cpu" then
      error(name .. " only supports the CPU device", 3)
    end
    dtype = opts.dtype
  end

  local result
  if row == 0 or col == 0 then
    result = torch.zeros(2, 0, { dtype = torch.int64 })
  else
    local rows = torch.arange(row, { dtype = torch.int64 }):reshape(row, 1)
    local columns = torch.arange(col, { dtype = torch.int64 }):reshape(1, col)
    local offsets = columns:sub(rows)
    local mask
    if keep_lower then
      mask = offsets:le(diagonal)
    else
      mask = offsets:ge(diagonal)
    end
    result = mask:nonzero():t():contiguous()
  end
  if dtype ~= nil then
    result = result:to(dtype)
  end
  return result
end

function torch.tril_indices(row, col, diagonal, opts)
  return triangular_indices(row, col, diagonal, opts, true, "tril_indices")
end

function torch.triu_indices(row, col, diagonal, opts)
  return triangular_indices(row, col, diagonal, opts, false, "triu_indices")
end

function torch.trace(a)
  if not Tensor.is_tensor(a) then
    error("torch.trace expects a tensor", 2)
  end
  return a:trace()
end

function torch.diag(a, diagonal)
  if not Tensor.is_tensor(a) then
    error("torch.diag expects a tensor", 2)
  end
  return a:diag(diagonal)
end

function torch.diag_embed(a, diagonal)
  if not Tensor.is_tensor(a) then
    error("torch.diag_embed expects a tensor", 2)
  end
  return a:diag_embed(diagonal)
end

function torch.diagflat(a, diagonal)
  if not Tensor.is_tensor(a) then
    error("torch.diagflat expects a tensor", 2)
  end
  return a:reshape(-1):diag(diagonal)
end

function torch.vander(x, N, increasing)
  if not Tensor.is_tensor(x) then
    error("torch.vander expects a tensor", 2)
  end
  if x:dim() ~= 1 then
    error("torch.vander expects a 1-D tensor", 2)
  end
  if increasing ~= nil and type(increasing) ~= "boolean" then
    error("torch.vander increasing must be a boolean", 2)
  end
  return x:vander(N, increasing == true)
end

function torch.cross(a, b, dim)
  if not Tensor.is_tensor(a) or not Tensor.is_tensor(b) then
    error("torch.cross expects two tensors", 2)
  end
  return a:cross(b, dim)
end

function torch.mv(matrix, vector)
  return matrix:mv(vector)
end

function torch.outer(a, b)
  return a:outer(b)
end

function torch.kron(a, b)
  if not Tensor.is_tensor(a) then
    error("torch.kron expects a tensor as its first argument", 2)
  end
  return a:kron(b)
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

torch.special = {
  erf = torch.erf,
  erfc = torch.erfc,
  erfinv = torch.erfinv,
  i0 = torch.i0,
  i0e = torch.i0e,
  i1 = torch.i1,
  i1e = torch.i1e,
  lgamma = torch.lgamma,
  digamma = torch.digamma,
  psi = torch.digamma,
  ndtr = torch.ndtr,
  log_ndtr = torch.log_ndtr,
  ndtri = torch.ndtri,
  entr = torch.entr,
  xlogy = torch.xlogy,
  xlog1py = torch.xlog1py,
  expit = torch.sigmoid,
  logit = torch.logit,
}

return torch
