--
-- luatorch / tensor.lua
--
-- The Tensor class: a thin, idiomatic Lua layer over the at::Tensor handles
-- produced by libluatorch.so.  Conventions:
--
--   * dimensions and indices are 1-based (-1 = last), like Torch7
--   * mutating methods end in `_` and return self
--   * non-mutating methods return a new Tensor
--   * operators (+ - * / ^, unary -) are non-mutating
--   * reductions over no dimension (dim == nil) return a 0-dim tensor
--
local libmod = require("torch.lib")
local ffi = libmod.ffi
local lib = libmod.lib
local dtype_names = libmod.dtype_names

local ALL_DIMS = -1000

local profiler = require("torch.profiler")
local profiled = {}

local Tensor = {}
local mt = {}

local dtype_ids = {
  float32 = 0, float = 0,
  float64 = 1, double = 1,
  int64 = 2, long = 2,
  int32 = 3, int = 3,
  int16 = 4, short = 4,
  int8 = 5, char = 5,
  uint8 = 6, byte = 6,
  bool = 7, boolean = 7,
  float16 = 8, half = 8,
}

local ctypes = {
  [0] = "float",
  [1] = "double",
  [2] = "int64_t",
  [3] = "int32_t",
  [4] = "int16_t",
  [5] = "int8_t",
  [6] = "uint8_t",
  [7] = "uint8_t",
  [8] = "uint16_t",
}

local type_names = {
  [0] = "torch.FloatTensor",
  [1] = "torch.DoubleTensor",
  [2] = "torch.LongTensor",
  [3] = "torch.IntTensor",
  [4] = "torch.ShortTensor",
  [5] = "torch.CharTensor",
  [6] = "torch.ByteTensor",
  [7] = "torch.BoolTensor",
  [8] = "torch.HalfTensor",
}

local function last_error()
  return ffi.string(lib.lt_last_error())
end

local function check(status)
  if status ~= 0 then
    error(last_error(), 3)
  end
end

-- LuaJIT's GC does not know how much memory lives on the C++ side.  Without
-- accounting, out-of-place results pile up until a Lua-triggered collection and
-- C++ allocations become several times slower than PyTorch's refcounted free.
-- Track tensor bytes and force a full collection at a configurable threshold.
local external_bytes = 0
local gc_threshold = tonumber(os.getenv("LUATORCH_GC_THRESHOLD_MB") or "64") * 1024 * 1024

local function release(pointer)
  -- Do not call metadata helpers here: they clear the thread-local error
  -- message, and finalizers can run inside ffi.string() while Lua is trying to
  -- read that message.
  external_bytes = external_bytes - tonumber(lib.lt_free_counted(pointer))
end

local function wrap(pointer)
  if pointer == nil then
    error(last_error(), 3)
  end
  external_bytes = external_bytes + tonumber(lib.lt_nbytes(pointer))
  local handle = ffi.gc(pointer, release)
  if external_bytes > gc_threshold then
    collectgarbage("collect")
  end
  return setmetatable({ h = handle }, mt)
end

local function is_tensor(value)
  return type(value) == "table" and getmetatable(value) == mt
end

local function resolve_dtype(dtype)
  if dtype == nil then
    return nil
  end
  if type(dtype) == "number" then
    if dtype < 0 or dtype > 8 or dtype % 1 ~= 0 then
      error("invalid dtype id: " .. tostring(dtype), 3)
    end
    return dtype
  end
  if type(dtype) == "string" then
    local id = dtype_ids[dtype]
    if id == nil then
      error("unknown dtype: '" .. dtype .. "'", 3)
    end
    return id
  end
  error("dtype must be a dtype id or a string", 3)
end

local function shape_to_cdata(shape)
  local n = #shape
  if n == 0 then
    return nil, 0
  end
  local array = ffi.new("int64_t[?]", n)
  for i = 1, n do
    local value = shape[i]
    if type(value) ~= "number" or value % 1 ~= 0 then
      error("tensor shape must contain integers, got " .. tostring(value), 3)
    end
    array[i - 1] = value
  end
  return array, n
end

-- A tensor of the same shape and dtype as `reference`, filled with `value`.
-- Shared by nan_to_num, the NaN-aware reductions, hypot, and diag_embed.
local function full_like_value(reference, value)
  local array, ndim = shape_to_cdata(reference:sizes())
  return wrap(lib.lt_full(ndim, array, value, reference:dtype_id()))
end

local function is_dtype_integral(dtype)
  return dtype ~= 0 and dtype ~= 1 and dtype ~= 8
end

local function is_dtype_floating(dtype)
  return dtype == 0 or dtype == 1 or dtype == 8
end

local OPTION_KEYS = { dtype = true, device = true, requires_grad = true }

local function is_options(value)
  if type(value) ~= "table" then
    return false
  end
  local any = false
  for key in pairs(value) do
    if not OPTION_KEYS[key] then
      return false
    end
    any = true
  end
  return any
end

local function parse_shape_opts(...)
  local args = { ... }
  local opts = {}
  if #args > 0 and is_options(args[#args]) then
    opts = table.remove(args)
  end
  local shape
  if #args == 1 and type(args[1]) == "table" then
    shape = args[1]
    if #shape == 0 and next(shape) ~= nil then
      error("invalid shape table", 3)
    end
  else
    shape = args
  end
  if #shape == 0 then
    error("a tensor shape is required", 3)
  end
  local checked = {}
  for i = 1, #shape do
    local value = shape[i]
    if type(value) ~= "number" or value % 1 ~= 0 then
      error("tensor shape must contain integers, got " .. tostring(value), 3)
    end
    checked[i] = value
  end
  return checked, opts
end

local function apply_requires_grad(tensor, opts)
  if opts and opts.requires_grad then
    tensor:requires_grad_(true)
  end
  return tensor
end

local function check_device(opts)
  if opts and opts.device and opts.device ~= "cpu" then
    error("only the CPU device is supported by this build (requested '" ..
      tostring(opts.device) .. "')", 3)
  end
end

local function create(kind, shape, opts, value)
  opts = opts or {}
  check_device(opts)
  local dtype = resolve_dtype(opts.dtype) or 0
  local array, ndim = shape_to_cdata(shape)
  local pointer
  if kind == "zeros" then
    pointer = lib.lt_zeros(ndim, array, dtype)
  elseif kind == "ones" then
    pointer = lib.lt_ones(ndim, array, dtype)
  elseif kind == "empty" then
    pointer = lib.lt_empty(ndim, array, dtype)
  elseif kind == "rand" then
    pointer = lib.lt_rand(ndim, array, dtype)
  elseif kind == "randn" then
    pointer = lib.lt_randn(ndim, array, dtype)
  elseif kind == "full" then
    pointer = lib.lt_full(ndim, array, value, dtype)
  else
    error("unknown tensor constructor: " .. tostring(kind), 3)
  end
  return apply_requires_grad(wrap(pointer), opts)
end

-- ---------------------------------------------------------------------------
-- construction from Lua values
-- ---------------------------------------------------------------------------

local function table_shape(value)
  if type(value) ~= "table" then
    return nil
  end
  local n = #value
  if n == 0 then
    return { 0 }
  end
  local first = value[1]
  if type(first) == "table" then
    local sub = table_shape(first)
    for i = 2, n do
      local other = table_shape(value[i])
      if not other or #other ~= #sub then
        error("non-rectangular nested table", 3)
      end
      for d = 1, #sub do
        if other[d] ~= sub[d] then
          error("non-rectangular nested table", 3)
        end
      end
    end
    local shape = { n }
    for i = 1, #sub do
      shape[#shape + 1] = sub[i]
    end
    return shape
  end
  for i = 1, n do
    local v = value[i]
    if type(v) == "table" then
      error("non-rectangular nested table", 3)
    end
    if type(v) ~= "number" and type(v) ~= "boolean" then
      error("tensor values must be numbers or booleans, got " .. type(v), 3)
    end
  end
  return { n }
end

local function flatten_into(value, out)
  if type(value) == "table" then
    for i = 1, #value do
      flatten_into(value[i], out)
    end
  else
    out[#out + 1] = value
  end
end

-- Lua has no syntactic distinction between 1 and 1.0, so numeric tables
-- always become float32 (the PyTorch default floating dtype).  Use
-- {dtype = torch.int64} / torch.LongTensor for integer tensors.
local function infer_dtype(values)
  local saw_boolean, saw_number = false, false
  for i = 1, #values do
    if type(values[i]) == "boolean" then
      saw_boolean = true
    else
      saw_number = true
    end
  end
  if saw_boolean and saw_number then
    error("cannot mix booleans and numbers in one tensor", 3)
  end
  return saw_boolean and 7 or 0
end

-- ATen owns the IEEE-754 half conversion (round-to-nearest-even, overflow to
-- infinity), so a float16 tensor built from Lua values is staged as float32 and
-- cast natively.  Lua never encodes half bits itself.
local function cast_float16(tensor, opts)
  local half = wrap(lib.lt_to_dtype(tensor.h, 8))
  return apply_requires_grad(half, opts)
end

local function from_flat(flat, shape, opts)
  opts = opts or {}
  check_device(opts)
  local dtype = resolve_dtype(opts.dtype)
  if dtype == nil then
    dtype = infer_dtype(flat)
  end
  if dtype == 8 then
    return cast_float16(from_flat(flat, shape, { dtype = 0 }), opts)
  end
  local ctype = ctypes[dtype]
  local n = #flat
  local buffer = ffi.new(ctype .. "[?]", math.max(n, 1))
  for i = 1, n do
    local value = flat[i]
    if dtype == 7 then
      buffer[i - 1] = value and 1 or 0
    elseif type(value) == "boolean" then
      buffer[i - 1] = value and 1 or 0
    else
      buffer[i - 1] = value
    end
  end
  local array, ndim = shape_to_cdata(shape)
  local tensor = wrap(lib.lt_from_data(buffer, ndim, array, dtype))
  return apply_requires_grad(tensor, opts)
end

local function from_table(value, opts)
  local shape = table_shape(value)
  if not shape then
    error("torch.tensor: expected a table, number or boolean", 3)
  end
  local flat = {}
  flatten_into(value, flat)
  return from_flat(flat, shape, opts)
end

local function from_scalar(value, opts)
  opts = opts or {}
  local dtype = resolve_dtype(opts.dtype)
  if dtype == nil then
    dtype = type(value) == "boolean" and 7 or 0
  end
  if dtype == 8 then
    return cast_float16(from_scalar(value, { dtype = 0 }), opts)
  end
  local ctype = ctypes[dtype]
  local buffer = ffi.new(ctype .. "[1]")
  if dtype == 7 or type(value) == "boolean" then
    buffer[0] = value and 1 or 0
  else
    buffer[0] = value
  end
  local tensor = wrap(lib.lt_from_data(buffer, 0, nil, dtype))
  return apply_requires_grad(tensor, opts)
end

-- from_blob shares memory with an FFI array or Lua string.  The returned tensor
-- retains `buffer` so the backing memory cannot be collected while the tensor
-- lives.  Strings are immutable: treat their tensors as read-only.
function Tensor.from_blob(buffer, shape, opts)
  opts = opts or {}
  check_device(opts)
  if type(buffer) ~= "string" and type(buffer) ~= "cdata" then
    error("torch.from_blob expects a string or FFI cdata buffer", 3)
  end
  if type(shape) ~= "table" or #shape == 0 then
    error("torch.from_blob expects a non-empty shape table", 3)
  end
  local dtype = resolve_dtype(opts.dtype)
  if dtype == nil then dtype = 0 end
  local count = 1
  for _, value in ipairs(shape) do
    if type(value) ~= "number" or value % 1 ~= 0 or value < 0 then
      error("torch.from_blob: shape entries must be non-negative integers", 3)
    end
    count = count * value
  end
  local element_size = ffi.sizeof(ctypes[dtype])
  local needed = count * element_size
  local available
  if type(buffer) == "string" then
    available = #buffer
  elseif tostring(ffi.typeof(buffer)):match("%*%s*>$") then
    -- A typed pointer carries no size; callers must supply opts.nbytes.
    available = tonumber(opts.nbytes)
  else
    local ok, bytes = pcall(ffi.sizeof, buffer)
    available = ok and bytes or tonumber(opts.nbytes)
  end
  if available == nil or available < needed then
    error(string.format(
      "torch.from_blob: buffer has %s bytes, need %d for shape[%s]",
      tostring(available), needed, table.concat(shape, ", ")), 3)
  end
  local pointer
  if type(buffer) == "string" then
    pointer = ffi.cast("const char*", buffer)
  else
    pointer = ffi.cast("void*", buffer)
  end
  local array, ndim = shape_to_cdata(shape)
  local tensor = wrap(lib.lt_from_blob(pointer, ndim, array, dtype))
  tensor._owner = buffer
  return apply_requires_grad(tensor, opts)
end

local function tensor_ctor(...)
  local args = { ... }
  if #args == 1 and type(args[1]) == "table" then
    return from_table(args[1], { dtype = 0 })
  end
  -- Torch7 semantics: torch.Tensor(5) is a 1-D tensor of length 5.
  -- Use torch.tensor(5) for a 0-dim scalar.
  local shape = parse_shape_opts(...)
  return create("empty", shape, { dtype = 0 })
end

-- ---------------------------------------------------------------------------
-- metadata
-- ---------------------------------------------------------------------------

function Tensor:_cdim(dim)
  if dim == nil or dim == ALL_DIMS then
    return ALL_DIMS
  end
  if type(dim) ~= "number" then
    error("dimension must be a number, got " .. type(dim), 3)
  end
  if dim == 0 then
    error("dimensions are 1-based; dimension 0 is invalid", 3)
  end
  if dim > 0 then
    return dim - 1
  end
  return dim
end

function Tensor:dim()
  return tonumber(lib.lt_dim(self.h))
end

Tensor.ndimension = Tensor.dim

function Tensor:numel()
  return tonumber(lib.lt_numel(self.h))
end

function Tensor:dtype()
  return dtype_names[lib.lt_dtype(self.h)]
end

function Tensor:dtype_id()
  return lib.lt_dtype(self.h)
end

function Tensor:device()
  return lib.lt_device(self.h) == 1 and "cuda" or "cpu"
end

function Tensor:is_cuda()
  return lib.lt_device(self.h) == 1
end

function Tensor:is_contiguous()
  return lib.lt_is_contiguous(self.h) == 1
end

function Tensor:is_floating_point()
  return lib.lt_is_floating_point(self.h) == 1
end

function Tensor:requires_grad()
  return lib.lt_requires_grad(self.h) == 1
end

function Tensor:element_size()
  return tonumber(lib.lt_element_size(self.h))
end

function Tensor:nbytes()
  return tonumber(lib.lt_nbytes(self.h))
end

function Tensor:sizes()
  local n = self:dim()
  local array = ffi.new("int64_t[?]", math.max(n, 1))
  check(lib.lt_sizes(self.h, array))
  local result = {}
  for i = 1, n do
    result[i] = tonumber(array[i - 1])
  end
  return result
end

Tensor.shape = Tensor.sizes

function Tensor:size(dim)
  if dim == nil then
    return self:sizes()
  end
  local value = lib.lt_size(self.h, self:_cdim(dim))
  if value < 0 then
    error(last_error(), 2)
  end
  return tonumber(value)
end

function Tensor:type()
  return type_names[self:dtype_id()]
end

-- ---------------------------------------------------------------------------
-- copies, dtype and autograd plumbing
-- ---------------------------------------------------------------------------

function Tensor:clone()
  return wrap(lib.lt_clone(self.h))
end

function Tensor:contiguous()
  return wrap(lib.lt_contiguous(self.h))
end

function Tensor:detach()
  return wrap(lib.lt_detach(self.h))
end

function Tensor:copy_(src)
  if not is_tensor(src) then
    error("copy_ expects a Tensor", 3)
  end
  check(lib.lt_copy_(self.h, src.h))
  return self
end

function Tensor:to(dtype)
  if dtype == nil then
    return self
  end
  if type(dtype) == "string" and (dtype == "cpu" or dtype:match("^cuda")) then
    if dtype ~= "cpu" then
      error("CUDA is not available in this build", 3)
    end
    return self
  end
  return wrap(lib.lt_to_dtype(self.h, resolve_dtype(dtype)))
end

Tensor.type_as = function(self, other)
  return self:to(other:dtype())
end

function Tensor:float()
  return self:to(0)
end

function Tensor:double()
  return self:to(1)
end

function Tensor:long()
  return self:to(2)
end

function Tensor:int()
  return self:to(3)
end

function Tensor:half()
  return self:to(8)
end

function Tensor:bool()
  return self:to(7)
end

function Tensor:requires_grad_(flag)
  check(lib.lt_set_requires_grad(self.h, (flag == nil or flag) and 1 or 0))
  return self
end

function Tensor:has_grad()
  return lib.lt_has_grad(self.h) == 1
end

function Tensor:grad()
  if not self:has_grad() then
    return nil
  end
  return wrap(lib.lt_grad(self.h))
end

function Tensor:backward(gradient)
  if gradient ~= nil and not is_tensor(gradient) then
    error("backward expects an optional gradient Tensor", 3)
  end
  check(lib.lt_backward(self.h, gradient and gradient.h or nil))
  return self
end

function Tensor:zero_grad()
  local grad = self:grad()
  if grad then
    grad:zero_()
  end
  return self
end

-- ---------------------------------------------------------------------------
-- element access
-- ---------------------------------------------------------------------------

local function to_native_index(value, what)
  if type(value) ~= "number" or value % 1 ~= 0 then
    error(what .. ": index must be an integer", 4)
  end
  if value == 0 then
    error(what .. ": index 0 is invalid (indices are 1-based)", 4)
  end
  return value > 0 and value - 1 or value
end

local function index_cdata(self, indices)
  local n = self:dim()
  if #indices ~= n then
    error(string.format("expected %d indices, got %d", n, #indices), 4)
  end
  local array = ffi.new("int64_t[?]", math.max(n, 1))
  for i = 1, n do
    array[i - 1] = to_native_index(indices[i], "get/set")
  end
  return array
end

-- LuaJIT NaN-boxes values in doubles, so a NaN payload produced by an ATen
-- kernel can read back as nil or a NULL cdata instead of a Lua number. Map any
-- such non-number (and every NaN payload) to the canonical Lua NaN so values
-- round-trip through get/totable/iter regardless of their bit pattern.
local function canonical_double(value)
  if value == nil or type(value) == "cdata" then
    return 0 / 0
  end
  if value ~= value then
    return 0 / 0
  end
  return value
end

function Tensor:get(...)
  local indices = { ... }
  local n = self:dim()
  local array = index_cdata(self, indices)
  local dtype = self:dtype_id()
  if dtype == 7 then
    local out = ffi.new("int[1]")
    check(lib.lt_item_bool(self.h, n, array, out))
    return out[0] ~= 0
  elseif is_dtype_integral(dtype) then
    local out = ffi.new("int64_t[1]")
    check(lib.lt_item_i64(self.h, n, array, out))
    return tonumber(out[0])
  else
    local out = ffi.new("double[1]")
    check(lib.lt_item_f64(self.h, n, array, out))
    return canonical_double(out[0])
  end
end

function Tensor:set(...)
  local args = { ... }
  if #args < 1 then
    error("set expects indices followed by a value", 3)
  end
  local value = table.remove(args)
  local n = self:dim()
  local array = index_cdata(self, args)
  local dtype = self:dtype_id()
  if dtype == 7 then
    check(lib.lt_set_item_f64(self.h, n, array, value and 1 or 0))
  elseif is_dtype_integral(dtype) then
    if type(value) ~= "number" then
      error("integer tensors require numeric values", 3)
    end
    check(lib.lt_set_item_i64(self.h, n, array, math.floor(value)))
  else
    check(lib.lt_set_item_f64(self.h, n, array, value))
  end
  return self
end

function Tensor:item()
  if self:numel() ~= 1 then
    error("item() requires a tensor with exactly one element", 3)
  end
  if self:dim() == 0 then
    return self:get()
  end
  local flat = self:flatten(1, self:dim())
  return flat:get(1)
end

function Tensor:select(dim, index)
  return wrap(lib.lt_select(self.h, self:_cdim(dim), to_native_index(index, "select")))
end

function Tensor:narrow(dim, start, length)
  local d = self:_cdim(dim)
  return wrap(lib.lt_narrow(self.h, d, to_native_index(start, "narrow"), length))
end

function Tensor:slice(dim, start, stop, step)
  local d = self:_cdim(dim)
  local n = self:size(dim)
  step = step or 1
  if step < 0 then
    error("slice: negative step is not supported; use flip()", 3)
  end
  local s = start or 1
  local e = stop or n
  if s < 0 then s = n + s + 1 end
  if e < 0 then e = n + e + 1 end
  s = math.max(1, math.min(n, s))
  e = math.min(n, e)
  if e < s then
    return wrap(lib.lt_narrow(self.h, d, math.min(s - 1, n), 0))
  end
  return wrap(lib.lt_slice(self.h, d, s - 1, e, step))
end

function Tensor:flip(dim)
  return wrap(lib.lt_flip(self.h, self:_cdim(dim)))
end

function Tensor:masked_fill(mask, value)
  if not is_tensor(mask) then
    error("masked_fill expects a boolean mask Tensor", 3)
  end
  return wrap(lib.lt_masked_fill(self.h, mask.h, value))
end

function Tensor:masked_select(mask)
  if not is_tensor(mask) then
    error("masked_select expects a boolean mask Tensor", 3)
  end
  return wrap(lib.lt_masked_select(self.h, mask.h))
end

-- `indices` are 1-based, like every other Lua-facing index in this module.
function Tensor:index_select(dim, indices)
  if not is_tensor(indices) then
    error("index_select expects an integer Tensor of indices", 3)
  end
  local zero_based = indices:to(2):sub(1)
  return wrap(lib.lt_index_select(self.h, self:_cdim(dim), zero_based.h))
end

-- Flattened 1-based gather: the index tensor's shape is the output shape and
-- its values address elements in row-major order. The selected values stay
-- differentiable through index_select.
function Tensor:take(index)
  if not is_tensor(index) then
    error("take expects an integer index tensor", 3)
  end
  local dtype = index:dtype_id()
  if dtype ~= 2 and dtype ~= 3 and dtype ~= 4 and dtype ~= 5 then
    error("take expects an int8/int16/int32/int64 index tensor", 3)
  end
  local selected = self:reshape(-1):index_select(1, index:reshape(-1))
  return selected:reshape(index:sizes())
end

function Tensor:index_select_batch(dim, indices_list)
  if type(indices_list) ~= "table" or #indices_list == 0 then
    error("index_select_batch expects a non-empty table of index tensors", 3)
  end
  local array = ffi.new("lt_tensor*[?]", #indices_list)
  local converted = {}
  for index = 1, #indices_list do
    local indices = indices_list[index]
    if not is_tensor(indices) then
      error("index_select_batch entry " .. index .. " is not a Tensor", 3)
    end
    converted[index] = indices:to(2):sub(1)
    array[index - 1] = converted[index].h
  end
  local outputs = ffi.new("lt_tensor*[?]", #indices_list)
  check(lib.lt_index_select_batch(self.h, self:_cdim(dim), array,
    #indices_list, outputs))
  local result = {}
  for index = 1, #indices_list do
    result[index] = wrap(outputs[index - 1])
  end
  return result
end

-- Coordinates are returned 1-based, ready for indexing_select or get().
function Tensor:nonzero()
  return wrap(lib.lt_nonzero(self.h)):to(2):add(1)
end

function Tensor:gather(dim, index)
  if not is_tensor(index) then
    error("gather expects an integer Tensor index", 3)
  end
  local zero_based = index:to(2):sub(1)
  return wrap(lib.lt_gather(self.h, self:_cdim(dim), zero_based.h))
end

function Tensor:take_along_dim(indices, dim)
  if not is_tensor(indices) then
    error("take_along_dim expects an integer Tensor", 3)
  end
  if indices:dtype() ~= "int64" and indices:dtype() ~= "int32" then
    error("take_along_dim indices must be integer", 3)
  end
  local d = self:_cdim(dim)
  if indices:dim() ~= self:dim() then
    error("take_along_dim indices must have the same rank as input", 3)
  end
  local sizes = self:sizes()
  local index_sizes = indices:sizes()
  for axis = 1, #sizes do
    if axis ~= d + 1 and sizes[axis] ~= index_sizes[axis] then
      error("take_along_dim index size mismatch on axis " .. axis, 3)
    end
  end
  return self:gather(dim, indices)
end

function Tensor:scatter(dim, index, source)
  if not is_tensor(index) or not is_tensor(source) then
    error("scatter expects index and source Tensors", 3)
  end
  local zero_based = index:to(2):sub(1)
  return wrap(lib.lt_scatter(self.h, self:_cdim(dim), zero_based.h, source.h))
end

function Tensor:sort(dim, descending)
  local d = self:_cdim(dim or self:dim())
  local values_pointer = ffi.new("lt_tensor*[1]")
  local indices_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_sort(self.h, d, descending and 1 or 0,
    values_pointer, indices_pointer))
  return wrap(values_pointer[0]), wrap(indices_pointer[0]):add(1)
end

-- Sort indices without materializing the values.
function Tensor:argsort(dim, descending)
  local _, indices = self:sort(dim, descending)
  return indices
end

-- n-th discrete difference along one dimension, using repeated first
-- differences over narrowed views so autograd flows through subtraction.
function Tensor:diff(n, dim)
  n = n == nil and 1 or n
  if type(n) ~= "number" or n ~= n or n % 1 ~= 0 or n < 0 or
      n == math.huge then
    error("diff n must be a non-negative integer", 3)
  end
  n = math.floor(n)
  if self:dtype() == "bool" then
    error("diff does not support bool tensors", 3)
  end
  local axis = dim == nil and self:dim() or dim
  local result = self
  if n == 0 then
    return self:clone()
  end
  for _ = 1, n do
    local length = result:size(axis)
    if length <= 1 then
      result = result:narrow(axis, 1, 0)
    else
      result = result:narrow(axis, 2, length - 1):sub(
        result:narrow(axis, 1, length - 1))
    end
  end
  return result
end

function Tensor:median(dim)
  if dim ~= nil then
    local values = self:sort(dim)
    local size = values:size(dim)
    local lower = math.floor((size + 1) / 2)
    local upper = math.floor((size + 2) / 2)
    local lower_values = values:narrow(dim, lower, 1)
    local upper_values = values:narrow(dim, upper, 1)
    return lower_values:add(upper_values):mul(0.5)
  end
  if self:numel() == 0 then error("median of an empty tensor", 2) end
  local values = self:reshape(-1):sort()
  local count = values:numel()
  local lower = math.floor((count + 1) / 2)
  local upper = math.floor((count + 2) / 2)
  return (values:get(lower) + values:get(upper)) / 2
end

-- k-th smallest value and its 1-based index along one dimension.
function Tensor:kthvalue(k, dim, keepdim)
  if type(k) ~= "number" or k ~= k or k % 1 ~= 0 or k <= 0 or
      k == math.huge then
    error("kthvalue k must be a positive integer", 3)
  end
  k = math.floor(k)
  local axis = dim == nil and self:dim() or dim
  if k > self:size(axis) then
    error("kthvalue k is out of range for the selected dimension", 3)
  end
  local values, indices = self:sort(axis)
  local value = values:narrow(axis, k, 1)
  local index = indices:narrow(axis, k, 1)
  if not keepdim then
    value = value:squeeze(axis)
    index = index:squeeze(axis)
  end
  return value, index
end

function Tensor:quantile(q, dim, keepdim)
  if type(q) ~= "number" or q ~= q or q < 0 or q > 1 then
    error("quantile q must be a number in [0, 1]", 3)
  end
  if dim == nil then
    local sorted = self:reshape(-1):sort()
    local n = sorted:numel()
    if n == 0 then error("quantile of an empty tensor", 3) end
    local position = q * (n - 1) + 1
    local lower = math.floor(position)
    local upper = math.min(n, lower + 1)
    local fraction = position - lower
    local a = sorted:get(lower)
    local b = sorted:get(upper)
    return a + (b - a) * fraction
  end
  local sorted = self:sort(dim)
  local n = sorted:size(dim)
  if n == 0 then error("quantile of an empty tensor", 3) end
  local position = q * (n - 1) + 1
  local lower = math.floor(position)
  local upper = math.min(n, lower + 1)
  local fraction = position - lower
  local a = sorted:narrow(dim, lower, 1)
  local b = sorted:narrow(dim, upper, 1)
  local value = a:mul(1 - fraction):add(b:mul(fraction))
  if not keepdim then value = value:squeeze(dim) end
  return value
end

function Tensor:topk(k, dim, largest, sorted_flag)
  if largest == nil then largest = true end
  if sorted_flag == nil then sorted_flag = true end
  local d = self:_cdim(dim or self:dim())
  local values_pointer = ffi.new("lt_tensor*[1]")
  local indices_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_topk(self.h, k, d, largest and 1 or 0, sorted_flag and 1 or 0,
    values_pointer, indices_pointer))
  return wrap(values_pointer[0]), wrap(indices_pointer[0]):add(1)
end

function Tensor:unique(sorted_flag, return_inverse, return_counts)
  if sorted_flag == nil then sorted_flag = true end
  local values_pointer = ffi.new("lt_tensor*[1]")
  local inverse_pointer = ffi.new("lt_tensor*[1]")
  local counts_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_unique(self.h,
    sorted_flag and 1 or 0,
    return_inverse and 1 or 0,
    return_counts and 1 or 0,
    values_pointer, inverse_pointer, counts_pointer))
  local values = wrap(values_pointer[0])
  local inverse = inverse_pointer[0] ~= nil and
    wrap(inverse_pointer[0]):add(1) or nil
  local counts = counts_pointer[0] ~= nil and wrap(counts_pointer[0]) or nil
  return values, inverse, counts
end

-- Collapse only adjacent equal elements. The input is treated as flattened;
-- optional inverse and counts match torch.unique_consecutive.
function Tensor:unique_consecutive(return_inverse, return_counts)
  local data = self:reshape(-1):totable()
  local values, inverse, counts = {}, {}, {}
  local current
  for i = 1, #data do
    local value = data[i]
    if i == 1 or value ~= current then
      current = value
      values[#values + 1] = value
      counts[#counts + 1] = 1
    else
      counts[#counts] = counts[#counts] + 1
    end
    inverse[i] = #values
  end
  local value_tensor = Tensor.from_table(values, { dtype = self:dtype_id() })
  local inverse_tensor = Tensor.from_table(inverse, { dtype = 2 })
  local count_tensor = Tensor.from_table(counts, { dtype = 2 })
  if return_inverse and return_counts then
    return value_tensor, inverse_tensor, count_tensor
  elseif return_inverse then
    return value_tensor, inverse_tensor
  elseif return_counts then
    return value_tensor, count_tensor
  end
  return value_tensor
end

-- ---------------------------------------------------------------------------
-- elementwise operators
-- ---------------------------------------------------------------------------

local function binary_method(name, tensor_fn, scalar_fn, inplace)
  local method = name .. (inplace and "_" or "")
  Tensor[method] = function(self, other)
    local pointer
    if is_tensor(other) then
      pointer = tensor_fn(self.h, other.h, inplace and 1 or 0)
    elseif type(other) == "number" and scalar_fn ~= nil then
      pointer = scalar_fn(self.h, other, inplace and 1 or 0)
    else
      error(method .. " expects a Tensor or a number", 3)
    end
    if inplace then
      if pointer == nil then
        error(last_error(), 3)
      end
      return self
    end
    return wrap(pointer)
  end
end

binary_method("add", lib.lt_add, lib.lt_add_scalar, false)
binary_method("add", lib.lt_add, lib.lt_add_scalar, true)
binary_method("sub", lib.lt_sub, lib.lt_sub_scalar, false)
binary_method("sub", lib.lt_sub, lib.lt_sub_scalar, true)
binary_method("mul", lib.lt_mul, lib.lt_mul_scalar, false)
binary_method("mul", lib.lt_mul, lib.lt_mul_scalar, true)
binary_method("div", lib.lt_div, lib.lt_div_scalar, false)
binary_method("div", lib.lt_div, lib.lt_div_scalar, true)
binary_method("pow", lib.lt_pow, lib.lt_pow_scalar, false)
binary_method("pow", lib.lt_pow, lib.lt_pow_scalar, true)
binary_method("remainder", lib.lt_remainder, lib.lt_remainder_scalar, false)
binary_method("remainder", lib.lt_remainder, lib.lt_remainder_scalar, true)
binary_method("maximum", lib.lt_maximum, nil, false)
binary_method("minimum", lib.lt_minimum, nil, false)

local function nan_extreme(self, other, name, is_maximum)
  if not is_tensor(other) then
    error(name .. " expects a tensor", 3)
  end
  if self:dtype_id() ~= other:dtype_id() then
    error(name .. " expects both tensors to share a dtype", 3)
  end
  if self:dtype() == "bool" then
    error(name .. " does not support bool tensors", 3)
  end
  local value
  if is_maximum then
    value = self:maximum(other)
  else
    value = self:minimum(other)
  end
  local self_nan, other_nan = self:isnan(), other:isnan()
  -- If either side is NaN, return the other side; if both are NaN the first
  -- selection returns the NaN from `other`.
  local chosen = wrap(lib.lt_where(other_nan.h, self.h, value.h))
  return wrap(lib.lt_where(self_nan.h, other.h, chosen.h))
end

function Tensor:fmax(other)
  return nan_extreme(self, other, "fmax", true)
end

function Tensor:fmin(other)
  return nan_extreme(self, other, "fmin", false)
end

-- Torch7 naming compatibility: cmul/cdiv are elementwise multiply/divide.
Tensor.cmul = Tensor.mul
Tensor.cmul_ = Tensor.mul_
Tensor.cdiv = Tensor.div
Tensor.cdiv_ = Tensor.div_

local function addc_operation(self, first, second, value, divide, name)
  if not is_tensor(first) or not is_tensor(second) then
    error(name .. " expects two tensors", 3)
  end
  if self:dtype() == "bool" or first:dtype() == "bool" or
      second:dtype() == "bool" then
    error(name .. " does not support bool tensors", 3)
  end
  value = value == nil and 1 or value
  if type(value) ~= "number" then
    error(name .. " value must be a number", 3)
  end
  local product
  if divide then
    product = first:div(second)
  else
    product = first:mul(second)
  end
  if value ~= 1 then
    product = product:mul(value)
  end
  return self:add(product)
end

function Tensor:addcmul(first, second, value)
  return addc_operation(self, first, second, value, false, "addcmul")
end

function Tensor:addcdiv(first, second, value)
  return addc_operation(self, first, second, value, true, "addcdiv")
end

-- Linear interpolation: self + weight * (other - self). The weight may be a
-- number or a tensor broadcast against the inputs; all operands are floating
-- point and share a dtype.
function Tensor:lerp(other, weight)
  if not is_tensor(other) then
    error("lerp expects an end tensor", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float16" and dtype ~= "float32" and dtype ~= "float64" then
    error("lerp supports float16, float32, and float64 tensors", 3)
  end
  if other:dtype() ~= dtype then
    error("lerp expects both inputs to share a dtype", 3)
  end
  if weight == nil then
    error("lerp expects a weight", 3)
  end
  if type(weight) ~= "number" and not is_tensor(weight) then
    error("lerp weight must be a number or a tensor", 3)
  end
  if is_tensor(weight) and weight:dtype() ~= dtype then
    error("lerp weight must match the input dtype", 3)
  end
  return self:add(other:sub(self):mul(weight))
end

local function float_binary_operand(self, other, name)
  if not is_tensor(other) then
    error(name .. " expects a tensor", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float16" and dtype ~= "float32" and dtype ~= "float64" then
    error(name .. " supports float16, float32, and float64 tensors", 3)
  end
  if other:dtype() ~= dtype then
    error(name .. " expects both tensors to share a dtype", 3)
  end
end

-- log(exp(self) + exp(other)), computed with a max shift so large inputs do
-- not overflow. Inputs broadcast and must share a floating dtype.
function Tensor:logaddexp(other)
  float_binary_operand(self, other, "logaddexp")
  local difference = self:sub(other):abs()
  local maximum = self:maximum(other)
  return maximum:add(difference:neg():exp():log1p())
end

-- log2(2^self + 2^other), using the same stable shift in base 2.
function Tensor:logaddexp2(other)
  float_binary_operand(self, other, "logaddexp2")
  local difference = self:sub(other):abs()
  local maximum = self:maximum(other)
  local ln2 = math.log(2)
  return maximum:add(difference:mul(-ln2):exp():log1p():div(ln2))
end

-- sqrt(self^2 + other^2), scaled by the larger magnitude so large finite
-- inputs do not overflow and infinite inputs stay infinite.
function Tensor:hypot(other)
  float_binary_operand(self, other, "hypot")
  local absolute_self = self:abs()
  local absolute_other = other:abs()
  local big = absolute_self:maximum(absolute_other)
  local small = absolute_self:minimum(absolute_other)
  local zeros = full_like_value(self, 0)
  local ratio = small:div(big)
  local special = big:eq(0):logical_or(big:isinf())
  ratio = wrap(lib.lt_where(special.h, zeros.h, ratio.h))
  local result = big:mul(ratio:mul(ratio):add(1):sqrt())
  -- C99 hypot(inf, nan) is inf; make the same choice when one magnitude is
  -- infinite even though the maximum propagates NaN.
  local infinite = self:isinf():logical_or(other:isinf())
  return wrap(lib.lt_where(infinite.h,
    full_like_value(self, math.huge).h, result.h))
end

-- Fractional part: self - trunc(self). Integer inputs return zero.
function Tensor:frac()
  if self:dtype() == "bool" then
    error("frac does not support bool tensors", 3)
  end
  return self:sub(self:trunc())
end

local function activation_dtype(tensor, name)
  local dtype = tensor:dtype()
  if dtype ~= "float16" and dtype ~= "float32" and dtype ~= "float64" then
    error(name .. " supports float16, float32, and float64 tensors", 3)
  end
end

-- Softplus: log(1 + exp(beta * x)) / beta. The max-shift form keeps large
-- positive and negative inputs finite, and threshold switches to the linear
-- branch above the cutoff.
function Tensor:softplus(beta, threshold)
  activation_dtype(self, "softplus")
  beta = beta == nil and 1 or beta
  threshold = threshold == nil and 20 or threshold
  if type(beta) ~= "number" or beta ~= beta or beta <= 0 or
      beta == math.huge then
    error("softplus beta must be a positive finite number", 3)
  end
  if type(threshold) ~= "number" or threshold ~= threshold or
      threshold == math.huge or threshold == -math.huge then
    error("softplus threshold must be a finite number", 3)
  end
  local scaled = self:mul(beta)
  local stable = scaled:relu():add(
    scaled:abs():neg():exp():log1p()):div(beta)
  local linear = scaled:gt(threshold)
  return wrap(lib.lt_where(linear.h, self.h, stable.h))
end

-- Exponential linear unit: x for x > 0, alpha * (exp(x) - 1) otherwise.
function Tensor:elu(alpha)
  activation_dtype(self, "elu")
  alpha = alpha == nil and 1 or alpha
  if type(alpha) ~= "number" or alpha ~= alpha then
    error("elu alpha must be a number", 3)
  end
  local negative = self:neg():relu():neg()
  local negative_branch = negative:expm1():mul(alpha)
  return wrap(lib.lt_where(self:gt(0).h, self.h, negative_branch.h))
end

-- Scaled exponential linear unit.
function Tensor:selu()
  activation_dtype(self, "selu")
  local alpha = 1.6732632423543772
  local scale = 1.0507009873554805
  return self:elu(alpha):mul(scale)
end

-- Sigmoid linear unit (swish): x * sigmoid(x).
function Tensor:silu()
  activation_dtype(self, "silu")
  return self:mul(self:sigmoid())
end

-- Gaussian error linear unit. The default exact form uses erf; the optional
-- "tanh" form uses the standard tanh approximation.
function Tensor:gelu(approximate)
  activation_dtype(self, "gelu")
  approximate = approximate or "none"
  if approximate ~= "none" and approximate ~= "tanh" then
    error("gelu approximate must be 'none' or 'tanh'", 3)
  end
  if approximate == "tanh" then
    local cubed = self:mul(self):mul(self)
    local shifted = self:add(cubed:mul(0.044715))
    local inner = shifted:mul(math.sqrt(2 / math.pi))
    return self:mul(inner:tanh():add(1)):mul(0.5)
  end
  local scaled = self:div(math.sqrt(2))
  return self:mul(scaled:erf():add(1)):mul(0.5)
end

-- Inverse sigmoid: log(x / (1 - x)). With eps, the input is first clamped to
-- [eps, 1 - eps] so exact 0/1 inputs produce finite values.
function Tensor:logit(eps)
  activation_dtype(self, "logit")
  local input = self
  if eps ~= nil then
    if type(eps) ~= "number" or eps ~= eps or eps < 0 or eps >= 0.5 then
      error("logit eps must be a number in [0, 0.5)", 3)
    end
    input = self:clamp(eps, 1 - eps)
  end
  return input:div(input:neg():add(1)):log()
end

local function clamp_bound(tensor, value, name)
  activation_dtype(tensor, name)
  if type(value) ~= "number" or value ~= value then
    error(name .. " bound must be a number", 3)
  end
end

function Tensor:clamp_min(minimum)
  clamp_bound(self, minimum, "clamp_min")
  local threshold = full_like_value(self, minimum)
  local mask = self:lt(minimum)
  return wrap(lib.lt_where(mask.h, threshold.h, self.h))
end

function Tensor:clamp_min_(minimum)
  return self:copy_(self:clamp_min(minimum))
end

function Tensor:clamp_max(maximum)
  clamp_bound(self, maximum, "clamp_max")
  local threshold = full_like_value(self, maximum)
  local mask = self:gt(maximum)
  return wrap(lib.lt_where(mask.h, threshold.h, self.h))
end

function Tensor:clamp_max_(maximum)
  return self:copy_(self:clamp_max(maximum))
end

local function comparison_method(name, tensor_fn, scalar_fn)
  Tensor[name] = function(self, other)
    if is_tensor(other) then
      return wrap(tensor_fn(self.h, other.h))
    elseif type(other) == "number" then
      return wrap(scalar_fn(self.h, other))
    end
    error(name .. " expects a Tensor or a number", 3)
  end
end

comparison_method("eq", lib.lt_eq, lib.lt_eq_scalar)
comparison_method("ne", lib.lt_ne, lib.lt_ne_scalar)
comparison_method("lt", lib.lt_lt, lib.lt_lt_scalar)
comparison_method("le", lib.lt_le, lib.lt_le_scalar)
comparison_method("gt", lib.lt_gt, lib.lt_gt_scalar)
comparison_method("ge", lib.lt_ge, lib.lt_ge_scalar)

function Tensor:logical_and(other)
  return wrap(lib.lt_logical_and(self.h, other.h))
end

function Tensor:logical_or(other)
  return wrap(lib.lt_logical_or(self.h, other.h))
end

function Tensor:logical_not()
  return wrap(lib.lt_logical_not(self.h))
end

-- NaN / Inf predicates; the result is a bool tensor shaped like the input.
function Tensor:isnan()
  return wrap(lib.lt_isnan(self.h))
end

function Tensor:isinf()
  return wrap(lib.lt_isinf(self.h))
end

function Tensor:isfinite()
  return wrap(lib.lt_isfinite(self.h))
end

function Tensor:isposinf()
  return self:isinf():logical_and(self:gt(0))
end

function Tensor:isneginf()
  return self:isinf():logical_and(self:lt(0))
end

local nan_to_num_limits = {
  float16 = { posinf = 65504.0, neginf = -65504.0 },
  float32 = {
    posinf = 3.4028234663852886e38,
    neginf = -3.4028234663852886e38,
  },
  float64 = {
    posinf = 1.7976931348623157e308,
    neginf = -1.7976931348623157e308,
  },
}

-- Replace NaN, +Inf, and -Inf with finite values. Floating tensors default to
-- NaN -> 0 and infinities -> the largest/smallest finite value of the dtype.
-- Integer and bool tensors have no non-finite values and are returned as a
-- clone. Replaced entries contribute zero gradient, matching PyTorch.
function Tensor:nan_to_num(nan, posinf, neginf)
  local arguments = { { "nan", nan }, { "posinf", posinf },
    { "neginf", neginf } }
  for i = 1, #arguments do
    local value = arguments[i][2]
    if value ~= nil and type(value) ~= "number" then
      error("nan_to_num " .. arguments[i][1] .. " must be a number", 3)
    end
  end
  local limits = nan_to_num_limits[self:dtype()]
  if limits == nil then
    return self:clone()
  end
  nan = nan == nil and 0.0 or nan
  posinf = posinf == nil and limits.posinf or posinf
  neginf = neginf == nil and limits.neginf or neginf

  local result = self
  local mask = result:isnan()
  local replacement = full_like_value(self, nan)
  result = wrap(lib.lt_where(mask.h, replacement.h, result.h))

  mask = result:isinf():logical_and(result:gt(0))
  replacement = full_like_value(self, posinf)
  result = wrap(lib.lt_where(mask.h, replacement.h, result.h))

  mask = result:isinf():logical_and(result:lt(0))
  replacement = full_like_value(self, neginf)
  result = wrap(lib.lt_where(mask.h, replacement.h, result.h))
  return result
end

local unary_fns = {
  neg = lib.lt_neg, abs = lib.lt_abs, exp = lib.lt_exp,
  expm1 = lib.lt_expm1, log = lib.lt_log, log1p = lib.lt_log1p,
  log2 = lib.lt_log2, log10 = lib.lt_log10, erf = lib.lt_erf,
  erfc = lib.lt_erfc, sinc = lib.lt_sinc, sqrt = lib.lt_sqrt,
  rsqrt = lib.lt_rsqrt,
  sin = lib.lt_sin, cos = lib.lt_cos, tan = lib.lt_tan, asin = lib.lt_asin,
  acos = lib.lt_acos, atan = lib.lt_atan, sinh = lib.lt_sinh,
  cosh = lib.lt_cosh, tanh = lib.lt_tanh, sigmoid = lib.lt_sigmoid,
  relu = lib.lt_relu, floor = lib.lt_floor, ceil = lib.lt_ceil,
  round = lib.lt_round, trunc = lib.lt_trunc, sign = lib.lt_sign,
  square = lib.lt_square, reciprocal = lib.lt_reciprocal,
}

for name, fn in pairs(unary_fns) do
  Tensor[name] = function(self)
    return wrap(fn(self.h, 0))
  end
  Tensor[name .. "_"] = function(self)
    local pointer = fn(self.h, 1)
    if pointer == nil then
      error(last_error(), 3)
    end
    return self
  end
end

function Tensor:clamp(minimum, maximum)
  if minimum == nil or maximum == nil then
    error("clamp expects a minimum and a maximum", 3)
  end
  return wrap(lib.lt_clamp(self.h, minimum, maximum, 0))
end

function Tensor:clamp_(minimum, maximum)
  if minimum == nil or maximum == nil then
    error("clamp_ expects a minimum and a maximum", 3)
  end
  if lib.lt_clamp(self.h, minimum, maximum, 1) == nil then
    error(last_error(), 3)
  end
  return self
end

function Tensor:fill_(value)
  if lib.lt_fill(self.h, value, 1) == nil then
    error(last_error(), 3)
  end
  return self
end

function Tensor:zero_()
  return self:fill_(0)
end

function Tensor:uniform_(low, high)
  low = low or 0
  high = high or 1
  check(lib.lt_uniform_(self.h, low, high))
  return self
end

function Tensor:normal_(mean, std)
  check(lib.lt_normal_(self.h,
    mean == nil and 0 or mean,
    std == nil and 1 or std))
  return self
end

function Tensor:exponential_(lambd)
  check(lib.lt_exponential_(self.h, lambd == nil and 1 or lambd))
  return self
end

function Tensor:bernoulli_(p)
  p = p == nil and 0.5 or p
  if p < 0 or p > 1 then
    error("bernoulli_: p must be between 0 and 1", 3)
  end
  check(lib.lt_bernoulli_(self.h, p))
  return self
end

function Tensor:multinomial(num_samples, replacement)
  -- ATen samples 0-based categories; expose 1-based indices in Lua.
  return wrap(lib.lt_multinomial(self.h, num_samples, replacement and 1 or 0)):add(1)
end

-- ---------------------------------------------------------------------------
-- linear algebra
-- ---------------------------------------------------------------------------

function Tensor:matmul(other)
  return wrap(lib.lt_matmul(self.h, other.h))
end

function Tensor:mm(other)
  return wrap(lib.lt_mm(self.h, other.h))
end

function Tensor:bmm(other)
  return wrap(lib.lt_bmm(self.h, other.h))
end

function Tensor:dot(other)
  return wrap(lib.lt_dot(self.h, other.h))
end

-- Linear algebra ------------------------------------------------------------

local function zeros_tensor(shape, dtype)
  local array, ndim = shape_to_cdata(shape)
  return wrap(lib.lt_zeros(ndim, array, dtype))
end

local function arange_tensor(start, stop, dtype)
  return wrap(lib.lt_arange(start, stop, 1, dtype))
end

local function stack_tensors(tensors, dim)
  local count = #tensors
  if count == 0 then
    error("cannot stack an empty tensor list", 3)
  end
  local array = ffi.new("lt_tensor*[?]", count)
  for i = 1, count do
    array[i - 1] = tensors[i].h
  end
  return wrap(lib.lt_stack(array, count, dim))
end

local function cat_tensors(tensors, dim)
  local count = #tensors
  if count == 0 then
    error("cannot concatenate an empty tensor list", 3)
  end
  local array = ffi.new("lt_tensor*[?]", count)
  for i = 1, count do
    array[i - 1] = tensors[i].h
  end
  return wrap(lib.lt_cat(array, count, dim))
end

-- Zero-based (row, column) offset grid, shape (rows, cols). Used to build
-- triangular and diagonal masks that broadcast over batch dimensions.
local function offset_grid(rows, cols)
  if rows == 0 or cols == 0 then
    return zeros_tensor({ rows, cols }, 2)
  end
  local row = arange_tensor(0, rows, 2):reshape(rows, 1)
  local column = arange_tensor(0, cols, 2):reshape(1, cols)
  return column:sub(row)
end

local function diagonal_offset(value, name)
  if value == nil then
    return 0
  end
  if type(value) ~= "number" or value ~= value or value % 1 ~= 0 or
      value == math.huge or value == -math.huge then
    error(name .. " diagonal must be an integer", 3)
  end
  return math.floor(value)
end

local function einstein_unary(tensor, equation)
  local array = ffi.new("lt_tensor*[?]", 1)
  array[0] = tensor.h
  return wrap(lib.lt_einsum(array, 1, equation))
end

local function integer_argument(value, name, minimum)
  if type(value) ~= "number" or value ~= value or value % 1 ~= 0 or
      value == math.huge or value == -math.huge then
    error(name .. " must be an integer", 3)
  end
  value = math.floor(value)
  if value < minimum then
    error(string.format("%s must be at least %d", name, minimum), 3)
  end
  return value
end

local function dimension_argument(value, rank, name)
  if type(value) ~= "number" or value ~= value or value % 1 ~= 0 or
      value == math.huge or value == -math.huge then
    error(name .. " must be an integer dimension", 3)
  end
  local dim = math.floor(value)
  if dim < 0 then
    dim = rank + dim + 1
  end
  if dim < 1 or dim > rank then
    error(string.format("%s dimension out of range (got %d, expected 1..%d)",
      name, value, rank), 3)
  end
  return dim
end

local lu_unpack_options = {
  unpack_data = true,
  unpack_pivots = true,
}

local function validate_pivot_values(values, m, what)
  for i = 1, #values do
    local value = values[i]
    if type(value) ~= "number" or value % 1 ~= 0 or value < 1 or value > m then
      error(string.format(
        "lu_unpack expects 1-based pivot indices in 1..%d at %s[%d]",
        m, what, i), 3)
    end
  end
end

local function validate_unpack_pivots(lu, pivots)
  if not is_tensor(pivots) then
    error("lu_unpack expects an int32 pivots tensor", 3)
  end
  if pivots:dtype() ~= "int32" then
    error("lu_unpack expects pivots with int32 dtype", 3)
  end
  local rank = lu:dim()
  if pivots:dim() ~= rank - 1 then
    error("lu_unpack expects pivots with one dimension less than LU", 3)
  end
  for dim = 1, rank - 2 do
    if pivots:size(dim) ~= lu:size(dim) then
      error("lu_unpack expects pivots to match the LU batch dimensions", 3)
    end
  end
  local m, n = lu:size(-2), lu:size(-1)
  local k = math.min(m, n)
  if pivots:size(-1) ~= k then
    error("lu_unpack pivots have the wrong length", 3)
  end
  local function walk(node, path)
    if type(node) ~= "table" or #node == 0 then return end
    if type(node[1]) == "table" then
      for i = 1, #node do
        walk(node[i], string.format("%s[%d]", path, i))
      end
    else
      validate_pivot_values(node, m, path)
    end
  end
  walk(pivots:totable(), "pivots")
  return k
end

local function permutation_matrix(values, m, k, dtype)
  local perm = {}
  for i = 1, m do perm[i] = i end
  for i = k, 1, -1 do
    local j = values[i]
    perm[i], perm[j] = perm[j], perm[i]
  end
  local matrix = zeros_tensor({ m, m }, dtype)
  for i = 1, m do
    matrix:set(i, perm[i], 1)
  end
  return matrix
end

-- Unpack the compact LU factorization returned by lu_factor into P, L, and U
-- with A = P L U. L and U are built with differentiable tensor arithmetic, so
-- gradients flow through the packed factor; P depends only on the integer
-- pivots and is detached, matching lu().
function Tensor:lu_unpack(pivots, opts)
  if self:dim() < 2 then
    error("lu_unpack expects a matrix or a batch of matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("lu_unpack supports float32 and float64 tensors", 3)
  end
  local k = validate_unpack_pivots(self, pivots)

  if opts == nil then
    opts = {}
  elseif type(opts) ~= "table" then
    error("lu_unpack options must be a table", 3)
  end
  for name in pairs(opts) do
    if not lu_unpack_options[name] then
      error("lu_unpack got unknown option '" .. tostring(name) .. "'", 3)
    end
  end
  for _, name in ipairs({ "unpack_data", "unpack_pivots" }) do
    if opts[name] ~= nil and type(opts[name]) ~= "boolean" then
      error("lu_unpack " .. name .. " must be a boolean", 3)
    end
  end

  local dtype_id = self:dtype_id()
  local m, n = self:size(-2), self:size(-1)
  local unpack_data = opts.unpack_data ~= false
  local unpack_pivots = opts.unpack_pivots ~= false

  local p
  if unpack_pivots then
    local batch_shape = {}
    for dim = 1, self:dim() - 2 do
      batch_shape[dim] = self:size(dim)
    end
    local batch_count = 1
    for i = 1, #batch_shape do
      batch_count = batch_count * batch_shape[i]
    end
    if batch_count == 0 then
      batch_shape[#batch_shape + 1] = m
      batch_shape[#batch_shape + 1] = m
      p = zeros_tensor(batch_shape, dtype_id)
    else
      local flat = pivots:reshape(batch_count, k)
      local parts = {}
      for batch = 1, batch_count do
        parts[batch] = permutation_matrix(
          flat:select(1, batch):totable(), m, k, dtype_id)
      end
      p = stack_tensors(parts, 0)
      batch_shape[#batch_shape + 1] = m
      batch_shape[#batch_shape + 1] = m
      p = p:reshape(batch_shape)
    end
  else
    p = zeros_tensor({ 0 }, dtype_id)
  end

  local l, u
  if unpack_data then
    local rows = arange_tensor(1, m + 1, dtype_id):unsqueeze(2)
    local cols = arange_tensor(1, k + 1, dtype_id):unsqueeze(1)
    local lower = rows:gt(cols):to(dtype_id)
    local diagonal = rows:eq(cols):to(dtype_id)
    l = self:narrow(-1, 1, k) * lower + diagonal

    rows = arange_tensor(1, k + 1, dtype_id):unsqueeze(2)
    cols = arange_tensor(1, n + 1, dtype_id):unsqueeze(1)
    local upper = rows:le(cols):to(dtype_id)
    u = self:narrow(-2, 1, k) * upper
  else
    l = zeros_tensor({ 0 }, dtype_id)
    u = zeros_tensor({ 0 }, dtype_id)
  end

  return p, l, u
end

function Tensor:inverse()
  return wrap(lib.lt_linalg_inv(self.h))
end

local inv_ex_options = {
  check_errors = true,
}

-- Error-returning inverse. Returns the inverse and an int32 info tensor:
-- zero entries mark successful inversions and positive entries identify a
-- singular batch element. check_errors=true raises instead, matching
-- torch.linalg.inv_ex.
function Tensor:inv_ex(opts)
  if self:dim() < 2 then
    error("inv_ex expects a matrix or a batch of matrices", 3)
  end
  if self:size(-2) ~= self:size(-1) then
    error("inv_ex expects square matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("inv_ex supports float32 and float64 tensors", 3)
  end
  if opts == nil then
    opts = {}
  elseif type(opts) ~= "table" then
    error("inv_ex options must be a table", 3)
  end
  for name in pairs(opts) do
    if not inv_ex_options[name] then
      error("inv_ex got unknown option '" .. tostring(name) .. "'", 3)
    end
  end
  if opts.check_errors ~= nil and type(opts.check_errors) ~= "boolean" then
    error("inv_ex check_errors must be a boolean", 3)
  end
  local inverse_pointer = ffi.new("lt_tensor*[1]")
  local info_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_linalg_inv_ex(self.h, opts.check_errors and 1 or 0,
    inverse_pointer, info_pointer))
  return wrap(inverse_pointer[0]), wrap(info_pointer[0])
end

-- Multiplicative inverse of tensordot. The first `ind` dimensions are grouped
-- into m rows and the remaining dimensions into n columns; the reshaped matrix
-- must be square. The inverse is reshaped to trailing dimensions first and
-- grouped dimensions last, matching torch.linalg.tensorinv.
function Tensor:tensorinv(ind)
  if self:dim() < 1 then
    error("tensorinv expects a tensor with at least one dimension", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("tensorinv supports float32 and float64 tensors", 3)
  end
  ind = ind == nil and 2 or ind
  if type(ind) ~= "number" or ind ~= ind or ind % 1 ~= 0 or
      ind == math.huge or ind == -math.huge then
    error("tensorinv ind must be an integer", 3)
  end
  ind = math.floor(ind)
  if ind < 1 or ind > self:dim() then
    error(string.format("tensorinv ind must be in 1..%d", self:dim()), 3)
  end
  local shape = self:sizes()
  local m, n = 1, 1
  for i = 1, ind do m = m * shape[i] end
  for i = ind + 1, #shape do n = n * shape[i] end
  if m ~= n then
    error(string.format(
      "tensorinv expects prod(shape[1:%d]) == prod(shape[%d:]), got %d and %d",
      ind, ind + 1, m, n), 3)
  end
  local output_shape = {}
  for i = ind + 1, #shape do output_shape[#output_shape + 1] = shape[i] end
  for i = 1, ind do output_shape[#output_shape + 1] = shape[i] end
  return self:reshape(m, n):inverse():reshape(output_shape)
end

-- Solve tensordot(A, X, X:dim()) = B. The leading B:dim() dimensions of A
-- are grouped into m rows and the remaining dimensions into n columns; the
-- system is solved as an m x n matrix equation. `dims` optionally moves the
-- listed dimensions of A to the end before grouping, matching
-- torch.linalg.tensorsolve.
function Tensor:tensorsolve(other, dims)
  if not is_tensor(other) then
    error("tensorsolve expects a tensor right-hand side", 3)
  end
  if self:dim() < 1 then
    error("tensorsolve expects a tensor with at least one dimension", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("tensorsolve supports float32 and float64 tensors", 3)
  end
  if other:dtype_id() ~= self:dtype_id() then
    error("tensorsolve expects the right-hand side to match the tensor dtype", 3)
  end
  local rank = self:dim()
  local rhs_rank = other:dim()
  if rhs_rank > rank then
    error("tensorsolve expects the right-hand side to have no more " ..
      "dimensions than A", 3)
  end

  local moved = self
  if dims ~= nil then
    if type(dims) ~= "table" then
      error("tensorsolve dims must be a table of dimensions", 3)
    end
    local axes, seen = {}, {}
    for i = 1, #dims do
      local axis = dimension_argument(dims[i], rank, "tensorsolve dims")
      if seen[axis] then
        error("tensorsolve dims must be unique", 3)
      end
      seen[axis] = true
      axes[i] = axis
    end
    if #axes > 0 then
      local permutation = {}
      for i = 1, rank do
        if not seen[i] then permutation[#permutation + 1] = i end
      end
      for i = 1, #axes do permutation[#permutation + 1] = axes[i] end
      moved = self:permute(unpack(permutation))
    end
  end

  local shape = moved:sizes()
  local m, n = 1, 1
  for i = 1, rhs_rank do m = m * shape[i] end
  for i = rhs_rank + 1, #shape do n = n * shape[i] end
  if m ~= n then
    error(string.format(
      "tensorsolve expects prod(shape[1:%d]) == prod(shape[%d:]), " ..
      "got %d and %d", rhs_rank, rhs_rank + 1, m, n), 3)
  end
  if other:numel() ~= m then
    error(string.format(
      "tensorsolve expects B to have %d elements, got %d",
      m, other:numel()), 3)
  end
  local solution = moved:reshape(m, n):solve(other:reshape(m))
  local output_shape = {}
  for i = rhs_rank + 1, #shape do
    output_shape[#output_shape + 1] = shape[i]
  end
  return solution:reshape(output_shape)
end

function Tensor:det()
  return wrap(lib.lt_linalg_det(self.h))
end

function Tensor:slogdet()
  local sign_pointer = ffi.new("lt_tensor*[1]")
  local logabsdet_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_linalg_slogdet(self.h, sign_pointer, logabsdet_pointer))
  return wrap(sign_pointer[0]), wrap(logabsdet_pointer[0])
end

function Tensor:solve(other, opts)
  if not is_tensor(other) then
    error("solve expects a tensor right-hand side", 3)
  end
  local left = true
  if opts ~= nil then
    if type(opts) ~= "table" then
      error("solve options must be a table", 3)
    end
    for name in pairs(opts) do
      if name ~= "left" then
        error("solve got unknown option '" .. tostring(name) .. "'", 3)
      end
    end
    if opts.left ~= nil then
      if type(opts.left) ~= "boolean" then
        error("solve left must be a boolean", 3)
      end
      left = opts.left
    end
  end
  if left then
    return wrap(lib.lt_linalg_solve(self.h, other.h))
  end
  -- X A = B  <=>  A^T X^T = B^T, so solve the transposed system and
  -- transpose the result's last two dimensions back.
  if self:dim() < 2 then
    error("solve with left=false expects a matrix or batch of matrices", 3)
  end
  if other:dim() < 2 then
    error("solve with left=false expects a right-hand side with at least " ..
      "two dimensions", 3)
  end
  local transposed_matrix = self:transpose(-2, -1)
  local transposed_rhs = other:transpose(-2, -1)
  local transposed_solution = wrap(
    lib.lt_linalg_solve(transposed_matrix.h, transposed_rhs.h))
  return transposed_solution:transpose(-2, -1)
end

local solve_ex_options = {
  left = true,
  check_errors = true,
}

-- Error-reporting dense solve. Returns the solution and an int32 `info`
-- tensor whose entries are zero on success or positive when the corresponding
-- batch element is singular. `left` defaults to true (A X = B); false solves
-- X A = B. `check_errors` defaults to false, matching
-- torch.linalg.solve_ex. Vector and matrix right-hand sides are accepted with
-- broadcast batch dimensions, and the solution is differentiable.
function Tensor:solve_ex(other, opts)
  if not is_tensor(other) then
    error("solve_ex expects a tensor right-hand side", 3)
  end
  if self:dim() < 2 or self:size(-2) ~= self:size(-1) then
    error("solve_ex expects square matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("solve_ex supports float32 and float64 tensors", 3)
  end
  if other:dtype_id() ~= self:dtype_id() then
    error("solve_ex expects the right-hand side to match the matrix dtype", 3)
  end
  if opts == nil then
    opts = {}
  elseif type(opts) ~= "table" then
    error("solve_ex options must be a table", 3)
  end
  for name in pairs(opts) do
    if not solve_ex_options[name] then
      error("solve_ex got unknown option '" .. tostring(name) .. "'", 3)
    end
  end
  for _, name in ipairs({ "left", "check_errors" }) do
    if opts[name] ~= nil and type(opts[name]) ~= "boolean" then
      error("solve_ex " .. name .. " must be a boolean", 3)
    end
  end
  local left = opts.left ~= false
  local n = self:size(-1)
  if left then
    if other:dim() == 1 then
      if other:size(1) ~= n then
        error("solve_ex right-hand side has the wrong length", 3)
      end
    elseif other:dim() >= 2 then
      if other:size(-2) ~= n then
        error("solve_ex right-hand side has the wrong number of rows", 3)
      end
    else
      error("solve_ex expects a vector or matrix right-hand side", 3)
    end
  else
    if other:dim() < 2 then
      error("solve_ex expects a right-hand side with at least 2 dimensions " ..
        "when left is false", 3)
    end
    if other:size(-1) ~= n then
      error("solve_ex right-hand side has the wrong number of columns", 3)
    end
  end
  local result_pointer = ffi.new("lt_tensor*[1]")
  local info_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_linalg_solve_ex(self.h, other.h, left and 1 or 0,
    opts.check_errors == true and 1 or 0, result_pointer, info_pointer))
  return wrap(result_pointer[0]), wrap(info_pointer[0])
end

local solve_triangular_options = {
  upper = true,
  left = true,
  unitriangular = true,
}

-- Solve A X = B (or X A = B when left is false) where A is triangular.
-- `upper` is required so a lower/upper mix-up cannot silently produce a
-- plausible-looking but wrong answer. Only float32/float64 are accepted:
-- CPU LAPACK has no half-precision triangular solve.
function Tensor:solve_triangular(other, opts)
  if not is_tensor(other) then
    error("solve_triangular expects a tensor right-hand side", 3)
  end
  if opts == nil then
    error("solve_triangular requires options with an upper boolean", 3)
  elseif type(opts) ~= "table" then
    error("solve_triangular options must be a table", 3)
  end
  for name in pairs(opts) do
    if not solve_triangular_options[name] then
      error("solve_triangular got unknown option '" .. tostring(name) .. "'", 3)
    end
  end
  if type(opts.upper) ~= "boolean" then
    error("solve_triangular upper must be a boolean", 3)
  end
  for _, name in ipairs({ "left", "unitriangular" }) do
    if opts[name] ~= nil and type(opts[name]) ~= "boolean" then
      error("solve_triangular " .. name .. " must be a boolean", 3)
    end
  end
  if self:dim() < 2 or self:size(-2) ~= self:size(-1) then
    error("solve_triangular expects square matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("solve_triangular supports float32 and float64 tensors", 3)
  end
  if other:dtype_id() ~= self:dtype_id() then
    error("solve_triangular expects the right-hand side to match the " ..
      "matrix dtype", 3)
  end
  local n = self:size(-1)
  local left = opts.left ~= false
  local upper = opts.upper
  local unitriangular = opts.unitriangular == true
  if other:dim() == 1 then
    if not left then
      error("solve_triangular requires a 2-D right-hand side when left " ..
        "is false", 3)
    end
    if other:size(1) ~= n then
      error("solve_triangular right-hand side has the wrong length", 3)
    end
    local solution = wrap(lib.lt_linalg_solve_triangular(
      self.h, other:unsqueeze(2).h, upper and 1 or 0, 1,
      unitriangular and 1 or 0))
    return solution:squeeze(2)
  end
  if other:dim() < 2 then
    error("solve_triangular expects a vector or matrix right-hand side", 3)
  end
  if left then
    if other:size(-2) ~= n then
      error("solve_triangular right-hand side has the wrong number of rows", 3)
    end
  elseif other:size(-1) ~= n then
    error("solve_triangular right-hand side has the wrong number of columns", 3)
  end
  return wrap(lib.lt_linalg_solve_triangular(
    self.h, other.h, upper and 1 or 0, left and 1 or 0,
    unitriangular and 1 or 0))
end

local lstsq_drivers = {
  gels = true,
  gelsy = true,
  gelsd = true,
  gelss = true,
}

function Tensor:lstsq(other, opts)
  if not is_tensor(other) then
    error("lstsq expects a tensor right-hand side", 3)
  end
  if opts == nil then
    opts = {}
  elseif type(opts) ~= "table" then
    error("lstsq options must be a table", 3)
  end
  local driver = opts.driver
  if driver ~= nil and not lstsq_drivers[driver] then
    error("lstsq driver must be one of gels, gelsy, gelsd, gelss", 3)
  end
  local rcond = opts.rcond
  if rcond ~= nil and (type(rcond) ~= "number" or rcond < 0) then
    error("lstsq rcond must be a non-negative number", 3)
  end
  local solution_pointer = ffi.new("lt_tensor*[1]")
  local residuals_pointer = ffi.new("lt_tensor*[1]")
  local rank_pointer = ffi.new("lt_tensor*[1]")
  local singular_values_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_linalg_lstsq(self.h, other.h, driver,
    rcond == nil and 0 or rcond, rcond == nil and 0 or 1,
    solution_pointer, residuals_pointer, rank_pointer,
    singular_values_pointer))
  return wrap(solution_pointer[0]), wrap(residuals_pointer[0]),
    wrap(rank_pointer[0]), wrap(singular_values_pointer[0])
end

function Tensor:cholesky(upper)
  return wrap(lib.lt_linalg_cholesky(self.h, upper and 1 or 0))
end

-- Inverse of a real symmetric positive-definite matrix from its Cholesky
-- factor. `self` is the lower factor L of A = L L^T by default; pass true for
-- the upper factor U of A = U^T U. Square and batched float32/float64 factors
-- are accepted, and the inverse is differentiable.
function Tensor:cholesky_inverse(upper)
  if self:dim() < 2 then
    error("cholesky_inverse expects a matrix or a batch of matrices", 3)
  end
  if self:size(-2) ~= self:size(-1) then
    error("cholesky_inverse expects square matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("cholesky_inverse supports float32 and float64 tensors", 3)
  end
  if upper ~= nil and type(upper) ~= "boolean" then
    error("cholesky_inverse upper must be a boolean", 3)
  end
  return wrap(lib.lt_linalg_cholesky_inverse(self.h, upper and 1 or 0))
end

-- Solve A X = B from the Cholesky factor of A. `self` is B with shape
-- (*, n, k); `other` is the lower factor L of A = L L^T by default, or the
-- upper factor U of A = U^T U when upper is true. Batch dimensions of B may
-- broadcast over the factor. Both operands must share float32/float64 dtype,
-- and the solution is differentiable.
function Tensor:cholesky_solve(other, upper)
  if not is_tensor(other) then
    error("cholesky_solve expects a Cholesky factor tensor", 3)
  end
  if self:dim() < 2 then
    error("cholesky_solve expects a right-hand side with at least 2 " ..
      "dimensions", 3)
  end
  if other:dim() < 2 then
    error("cholesky_solve expects a matrix or a batch of matrices", 3)
  end
  if other:size(-2) ~= other:size(-1) then
    error("cholesky_solve expects square Cholesky factors", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("cholesky_solve supports float32 and float64 tensors", 3)
  end
  if other:dtype_id() ~= self:dtype_id() then
    error("cholesky_solve expects the factor to match the right-hand side " ..
      "dtype", 3)
  end
  if self:size(-2) ~= other:size(-1) then
    error("cholesky_solve right-hand side has the wrong number of rows", 3)
  end
  if upper ~= nil and type(upper) ~= "boolean" then
    error("cholesky_solve upper must be a boolean", 3)
  end
  return wrap(lib.lt_linalg_cholesky_solve(self.h, other.h,
    upper and 1 or 0))
end

local cholesky_ex_options = {
  upper = true,
  check_errors = true,
}

-- Error-returning Cholesky factorization. Returns the factor and an int32
-- info tensor: info == 0 means success and a positive info reports the first
-- leading minor that is not positive-definite. With check_errors = true the
-- underlying LAPACK path raises instead, matching torch.linalg.cholesky_ex.
function Tensor:cholesky_ex(opts)
  if self:dim() < 2 then
    error("cholesky_ex expects a matrix or a batch of matrices", 3)
  end
  if self:size(-2) ~= self:size(-1) then
    error("cholesky_ex expects square matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("cholesky_ex supports float32 and float64 tensors", 3)
  end
  if opts == nil then
    opts = {}
  elseif type(opts) ~= "table" then
    error("cholesky_ex options must be a table", 3)
  end
  for name in pairs(opts) do
    if not cholesky_ex_options[name] then
      error("cholesky_ex got unknown option '" .. tostring(name) .. "'", 3)
    end
  end
  for _, name in ipairs({ "upper", "check_errors" }) do
    if opts[name] ~= nil and type(opts[name]) ~= "boolean" then
      error("cholesky_ex " .. name .. " must be a boolean", 3)
    end
  end
  local factor_pointer = ffi.new("lt_tensor*[1]")
  local info_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_linalg_cholesky_ex(self.h, opts.upper and 1 or 0,
    opts.check_errors and 1 or 0, factor_pointer, info_pointer))
  return wrap(factor_pointer[0]), wrap(info_pointer[0])
end

local function uplo_argument(uplo, name)
  if uplo == nil then return "L" end
  if type(uplo) ~= "string" then
    error(name .. " UPLO must be 'L' or 'U'", 3)
  end
  local normalized = uplo:upper()
  if normalized ~= "L" and normalized ~= "U" then
    error(name .. " UPLO must be 'L' or 'U'", 3)
  end
  return normalized
end

-- Eigenvalues and eigenvectors of real symmetric (or batched symmetric)
-- matrices. The lower triangle is used by default; pass "U" to read the
-- upper triangle. Eigenvalues are returned in ascending order and the
-- eigenvectors are the columns of the second result, matching eigh.
function Tensor:eigh(uplo)
  if self:dim() < 2 then
    error("eigh expects a matrix or a batch of matrices", 3)
  end
  if self:size(-2) ~= self:size(-1) then
    error("eigh expects square matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("eigh supports float32 and float64 tensors", 3)
  end
  local values_pointer = ffi.new("lt_tensor*[1]")
  local vectors_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_linalg_eigh(self.h, uplo_argument(uplo, "eigh"),
    values_pointer, vectors_pointer))
  return wrap(values_pointer[0]), wrap(vectors_pointer[0])
end

-- Eigenvalues only, through at::linalg_eigvalsh so the eigenvectors are not
-- materialized. Uses the same square/batch and dtype rules as eigh.
function Tensor:eigvalsh(uplo)
  if self:dim() < 2 then
    error("eigvalsh expects a matrix or a batch of matrices", 3)
  end
  if self:size(-2) ~= self:size(-1) then
    error("eigvalsh expects square matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("eigvalsh supports float32 and float64 tensors", 3)
  end
  return wrap(lib.lt_linalg_eigvalsh(self.h, uplo_argument(uplo, "eigvalsh")))
end

function Tensor:svd(full_matrices)
  if full_matrices == nil then full_matrices = true end
  local u_pointer = ffi.new("lt_tensor*[1]")
  local s_pointer = ffi.new("lt_tensor*[1]")
  local vh_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_linalg_svd(self.h, full_matrices and 1 or 0,
    u_pointer, s_pointer, vh_pointer))
  return wrap(u_pointer[0]), wrap(s_pointer[0]), wrap(vh_pointer[0])
end

-- Singular values only. This follows the same CPU LAPACK path as svd with
-- full_matrices=false, returns the descending values in a tensor of shape
-- (*, min(m, n)), and preserves autograd through the singular values.
function Tensor:svdvals()
  if self:dim() < 2 then
    error("svdvals expects a matrix or a batch of matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("svdvals supports float32 and float64 tensors", 3)
  end
  local _, singular_values = self:svd(false)
  return singular_values
end

function Tensor:qr(mode)
  mode = mode or "reduced"
  local q_pointer = ffi.new("lt_tensor*[1]")
  local r_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_linalg_qr(self.h, mode, q_pointer, r_pointer))
  return wrap(q_pointer[0]), wrap(r_pointer[0])
end

-- Reconstruct the first n columns of a product of Householder reflectors from
-- the compact representation produced by geqrf. A has shape (*, m, n) with
-- m >= n, and tau has shape (*, k) with k <= n and exactly A's batch
-- dimensions. The diagonal of A is ignored and each reflector uses an implied
-- leading one. Only real float32/float64 are accepted: the CPU LAPACK path
-- has no half-precision kernel.
function Tensor:householder_product(tau)
  if not is_tensor(tau) then
    error("householder_product expects a tau tensor", 3)
  end
  if self:dim() < 2 then
    error("householder_product expects a matrix or a batch of matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("householder_product supports float32 and float64 tensors", 3)
  end
  if tau:dtype_id() ~= self:dtype_id() then
    error("householder_product expects tau to match the input dtype", 3)
  end
  local m, n = self:size(-2), self:size(-1)
  if m < n then
    error("householder_product expects input.shape[-2] >= input.shape[-1]", 3)
  end
  if tau:dim() ~= self:dim() - 1 then
    error("householder_product expects tau to have one dimension less than " ..
      "the input", 3)
  end
  for dim = 1, self:dim() - 2 do
    if tau:size(dim) ~= self:size(dim) then
      error("householder_product expects tau batch dimensions to equal " ..
        "input batch dimensions", 3)
    end
  end
  if tau:size(-1) > n then
    error("householder_product expects tau:size(-1) <= input.shape[-1]", 3)
  end
  return wrap(lib.lt_linalg_householder_product(self.h, tau.h))
end

-- LU decomposition with partial pivoting. Returns P, L, U such that
-- A = P L U for square and rectangular matrices, with an optional batch
-- dimension. CPU LAPACK only implements the pivoted float32/float64 path:
-- other dtypes and the unpivoted variant are not exposed.
function Tensor:lu()
  if self:dim() < 2 then
    error("lu expects a matrix or a batch of matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("lu supports float32 and float64 tensors", 3)
  end
  local p_pointer = ffi.new("lt_tensor*[1]")
  local l_pointer = ffi.new("lt_tensor*[1]")
  local u_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_linalg_lu(self.h, p_pointer, l_pointer, u_pointer))
  return wrap(p_pointer[0]), wrap(l_pointer[0]), wrap(u_pointer[0])
end

-- Compact LU factorization with partial pivoting. Returns the packed LU
-- matrix and ATen's 1-based pivots, so the result can be reused by lu_solve.
-- A singular matrix raises instead of returning a factorization that would
-- divide by zero later. Only the float32/float64 CPU LAPACK path is exposed.
function Tensor:lu_factor()
  if self:dim() < 2 then
    error("lu_factor expects a matrix or a batch of matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("lu_factor supports float32 and float64 tensors", 3)
  end
  local lu_pointer = ffi.new("lt_tensor*[1]")
  local pivots_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_linalg_lu_factor(self.h, lu_pointer, pivots_pointer))
  return wrap(lu_pointer[0]), wrap(pivots_pointer[0])
end

local lu_factor_ex_options = {
  pivot = true,
  check_errors = true,
}

-- Compact LU factorization that reports singularity through an int32 `info`
-- tensor instead of raising. Returns the packed LU, 1-based pivots, and info;
-- positive info entries identify a zero U diagonal. `pivot` defaults to true
-- (the CPU LAPACK path has no unpivoted kernel) and `check_errors` defaults to
-- false, matching torch.linalg.lu_factor_ex. Square, rectangular, batched, and
-- empty float32/float64 inputs are accepted.
function Tensor:lu_factor_ex(opts)
  if self:dim() < 2 then
    error("lu_factor_ex expects a matrix or a batch of matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("lu_factor_ex supports float32 and float64 tensors", 3)
  end
  if opts == nil then
    opts = {}
  elseif type(opts) ~= "table" then
    error("lu_factor_ex options must be a table", 3)
  end
  for name in pairs(opts) do
    if not lu_factor_ex_options[name] then
      error("lu_factor_ex got unknown option '" .. tostring(name) .. "'", 3)
    end
  end
  for _, name in ipairs({ "pivot", "check_errors" }) do
    if opts[name] ~= nil and type(opts[name]) ~= "boolean" then
      error("lu_factor_ex " .. name .. " must be a boolean", 3)
    end
  end
  if opts.pivot == false then
    error("lu_factor_ex pivot=false is not supported on the CPU backend", 3)
  end
  local lu_pointer = ffi.new("lt_tensor*[1]")
  local pivots_pointer = ffi.new("lt_tensor*[1]")
  local info_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_linalg_lu_factor_ex(self.h, 1,
    opts.check_errors == true and 1 or 0, lu_pointer, pivots_pointer,
    info_pointer))
  return wrap(lu_pointer[0]), wrap(pivots_pointer[0]), wrap(info_pointer[0])
end

local lu_solve_options = {
  left = true,
  adjoint = true,
}

-- Solve A X = B (or X A = B when left is false) from the packed LU and
-- 1-based pivots returned by lu_factor. `adjoint` solves A^T X = B for
-- real tensors. Only float32/float64 inputs are accepted, B must be at
-- least 2-D (use a column matrix for a single right-hand side), and the
-- pivots must have int32 dtype and exactly the factorization's batch shape.
function Tensor:lu_solve(pivots, other, opts)
  if not is_tensor(pivots) then
    error("lu_solve expects an int32 pivots tensor", 3)
  end
  if not is_tensor(other) then
    error("lu_solve expects a tensor right-hand side", 3)
  end
  if opts == nil then
    opts = {}
  elseif type(opts) ~= "table" then
    error("lu_solve options must be a table", 3)
  end
  for name in pairs(opts) do
    if not lu_solve_options[name] then
      error("lu_solve got unknown option '" .. tostring(name) .. "'", 3)
    end
  end
  for _, name in ipairs({ "left", "adjoint" }) do
    if opts[name] ~= nil and type(opts[name]) ~= "boolean" then
      error("lu_solve " .. name .. " must be a boolean", 3)
    end
  end
  if self:dim() < 2 or self:size(-2) ~= self:size(-1) then
    error("lu_solve expects square matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("lu_solve supports float32 and float64 tensors", 3)
  end
  if pivots:dtype() ~= "int32" then
    error("lu_solve expects pivots with int32 dtype", 3)
  end
  if other:dtype_id() ~= self:dtype_id() then
    error("lu_solve expects the right-hand side to match the matrix dtype", 3)
  end
  local n = self:size(-1)
  if pivots:dim() ~= self:dim() - 1 then
    error("lu_solve expects pivots with one dimension less than the matrix", 3)
  end
  for dim = 1, self:dim() - 2 do
    if pivots:size(dim) ~= self:size(dim) then
      error("lu_solve expects pivots to match the matrix batch dimensions", 3)
    end
  end
  if pivots:size(-1) ~= n then
    error("lu_solve pivots have the wrong length", 3)
  end
  if other:dim() < 2 then
    error("lu_solve expects a right-hand side with at least 2 dimensions", 3)
  end
  local left = opts.left ~= false
  local adjoint = opts.adjoint == true
  if left then
    if other:size(-2) ~= n then
      error("lu_solve right-hand side has the wrong number of rows", 3)
    end
  elseif other:size(-1) ~= n then
    error("lu_solve right-hand side has the wrong number of columns", 3)
  end
  return wrap(lib.lt_linalg_lu_solve(self.h, pivots.h, other.h,
    left and 1 or 0, adjoint and 1 or 0))
end

local ldl_factor_options = {
  hermitian = true,
}

-- Compact LDL^T factorization of a real symmetric matrix or batch of
-- matrices. Returns `LD` with the same shape as the input and int32 1-based
-- `pivots` of shape (*, n). Following LAPACK sytrf, the unit lower-triangular
-- multipliers are stored below the diagonal and the diagonal blocks of D on
-- the diagonal. A singular matrix raises; use ldl_factor_ex to report it
-- through info. The CPU path supports float32/float64 only, and the
-- factorization has no implemented derivative.
function Tensor:ldl_factor(opts)
  if self:dim() < 2 then
    error("ldl_factor expects a matrix or a batch of matrices", 3)
  end
  if self:size(-2) ~= self:size(-1) then
    error("ldl_factor expects square matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("ldl_factor supports float32 and float64 tensors", 3)
  end
  if opts == nil then
    opts = {}
  elseif type(opts) ~= "table" then
    error("ldl_factor options must be a table", 3)
  end
  for name in pairs(opts) do
    if not ldl_factor_options[name] then
      error("ldl_factor got unknown option '" .. tostring(name) .. "'", 3)
    end
  end
  if opts.hermitian ~= nil and type(opts.hermitian) ~= "boolean" then
    error("ldl_factor hermitian must be a boolean", 3)
  end
  local ld_pointer = ffi.new("lt_tensor*[1]")
  local pivots_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_linalg_ldl_factor(self.h, opts.hermitian == true and 1 or 0,
    ld_pointer, pivots_pointer))
  return wrap(ld_pointer[0]), wrap(pivots_pointer[0])
end

local ldl_factor_ex_options = {
  hermitian = true,
  check_errors = true,
}

-- Error-reporting LDL^T factorization. Returns the compact `LD` factor, int32
-- 1-based `pivots`, and an int32 `info` tensor whose entries are zero on
-- success or positive when the corresponding batch element is singular.
-- `hermitian` defaults to false and `check_errors` defaults to false, matching
-- torch.linalg.ldl_factor_ex. Float32/float64 square, batched, and empty
-- inputs are accepted, and the factorization has no implemented derivative.
function Tensor:ldl_factor_ex(opts)
  if self:dim() < 2 then
    error("ldl_factor_ex expects a matrix or a batch of matrices", 3)
  end
  if self:size(-2) ~= self:size(-1) then
    error("ldl_factor_ex expects square matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("ldl_factor_ex supports float32 and float64 tensors", 3)
  end
  if opts == nil then
    opts = {}
  elseif type(opts) ~= "table" then
    error("ldl_factor_ex options must be a table", 3)
  end
  for name in pairs(opts) do
    if not ldl_factor_ex_options[name] then
      error("ldl_factor_ex got unknown option '" .. tostring(name) .. "'", 3)
    end
  end
  for _, name in ipairs({ "hermitian", "check_errors" }) do
    if opts[name] ~= nil and type(opts[name]) ~= "boolean" then
      error("ldl_factor_ex " .. name .. " must be a boolean", 3)
    end
  end
  local ld_pointer = ffi.new("lt_tensor*[1]")
  local pivots_pointer = ffi.new("lt_tensor*[1]")
  local info_pointer = ffi.new("lt_tensor*[1]")
  check(lib.lt_linalg_ldl_factor_ex(self.h,
    opts.hermitian == true and 1 or 0,
    opts.check_errors == true and 1 or 0, ld_pointer, pivots_pointer,
    info_pointer))
  return wrap(ld_pointer[0]), wrap(pivots_pointer[0]), wrap(info_pointer[0])
end

local ldl_solve_options = {
  hermitian = true,
}

-- Solve A X = B from the compact LDL^T factor and 1-based pivots returned by
-- ldl_factor or ldl_factor_ex. B has shape (*, n, k); a single right-hand side
-- is a column matrix. Batch dimensions of B may broadcast over the factor,
-- both operands must share float32/float64 dtype, and the integer pivots must
-- match the factor's batch shape. The CPU path has no implemented derivative
-- for this solve.
function Tensor:ldl_solve(pivots, other, opts)
  if not is_tensor(pivots) then
    error("ldl_solve expects an int32 pivots tensor", 3)
  end
  if not is_tensor(other) then
    error("ldl_solve expects a tensor right-hand side", 3)
  end
  if self:dim() < 2 or self:size(-2) ~= self:size(-1) then
    error("ldl_solve expects square matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("ldl_solve supports float32 and float64 tensors", 3)
  end
  if pivots:dtype() ~= "int32" then
    error("ldl_solve expects pivots with int32 dtype", 3)
  end
  if other:dtype_id() ~= self:dtype_id() then
    error("ldl_solve expects the right-hand side to match the matrix dtype", 3)
  end
  local n = self:size(-1)
  if pivots:dim() ~= self:dim() - 1 then
    error("ldl_solve expects pivots with one dimension less than the matrix", 3)
  end
  for dim = 1, self:dim() - 2 do
    if pivots:size(dim) ~= self:size(dim) then
      error("ldl_solve expects pivots to match the matrix batch dimensions", 3)
    end
  end
  if pivots:size(-1) ~= n then
    error("ldl_solve pivots have the wrong length", 3)
  end
  if other:dim() < 2 then
    error("ldl_solve expects a right-hand side with at least 2 dimensions", 3)
  end
  if other:size(-2) ~= n then
    error("ldl_solve right-hand side has the wrong number of rows", 3)
  end
  if opts == nil then
    opts = {}
  elseif type(opts) ~= "table" then
    error("ldl_solve options must be a table", 3)
  end
  for name in pairs(opts) do
    if not ldl_solve_options[name] then
      error("ldl_solve got unknown option '" .. tostring(name) .. "'", 3)
    end
  end
  if opts.hermitian ~= nil and type(opts.hermitian) ~= "boolean" then
    error("ldl_solve hermitian must be a boolean", 3)
  end
  return wrap(lib.lt_linalg_ldl_solve(self.h, pivots.h, other.h,
    opts.hermitian == true and 1 or 0))
end

local pinv_options = {
  atol = true,
  rtol = true,
}

-- Moore-Penrose pseudoinverse through ATen's SVD-based `linalg_pinv`.
-- Unlike `inverse`, it accepts rectangular, rank-deficient, and batched
-- matrices; `atol`/`rtol` set the singular-value cutoff. Only the real
-- float32/float64 CPU LAPACK path is exposed, since half precision has no
-- SVD kernel here and luatorch has no complex dtype.
function Tensor:pinv(opts)
  if self:dim() < 2 then
    error("pinv expects a matrix or a batch of matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("pinv supports float32 and float64 tensors", 3)
  end
  if opts == nil then
    opts = {}
  elseif type(opts) ~= "table" then
    error("pinv options must be a table", 3)
  end
  for name in pairs(opts) do
    if not pinv_options[name] then
      error("pinv got unknown option '" .. tostring(name) .. "'", 3)
    end
  end
  local atol, rtol = opts.atol, opts.rtol
  if atol ~= nil and (type(atol) ~= "number" or atol ~= atol or atol < 0) then
    error("pinv atol must be a non-negative number", 3)
  end
  if rtol ~= nil and (type(rtol) ~= "number" or rtol ~= rtol or rtol < 0) then
    error("pinv rtol must be a non-negative number", 3)
  end
  return wrap(lib.lt_linalg_pinv(self.h, atol or 0, rtol or 0,
    atol ~= nil and 1 or 0, rtol ~= nil and 1 or 0))
end

-- Matrix exponential through ATen's `linalg_matrix_exp` (scaling and
-- squaring). The input is a square matrix or a batch of square matrices;
-- the output has the same shape and dtype. Only real float32/float64 are
-- exposed: ATen accepts float16, but its CPU half kernel is not numerically
-- reliable (small inputs can return NaN or large negative values), so half
-- is rejected instead of returning garbage.
function Tensor:matrix_exp()
  if self:dim() < 2 then
    error("matrix_exp expects a matrix or a batch of matrices", 3)
  end
  if self:size(-2) ~= self:size(-1) then
    error("matrix_exp expects square matrices", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float32" and dtype ~= "float64" then
    error("matrix_exp supports float32 and float64 tensors", 3)
  end
  return wrap(lib.lt_linalg_matrix_exp(self.h))
end

-- Integer matrix powers for square and batched square matrices. Binary
-- exponentiation keeps the autograd graph shallow; a negative exponent
-- inverts first, and exponent zero returns a fresh identity.
function Tensor:matrix_power(exponent)
  if type(exponent) ~= "number" or exponent ~= exponent or
      exponent % 1 ~= 0 or exponent == math.huge or exponent == -math.huge then
    error("matrix_power expects an integer exponent", 3)
  end
  if self:dim() < 2 or self:size(-2) ~= self:size(-1) then
    error("matrix_power expects square matrices", 3)
  end
  if self:dtype() == "bool" then
    error("matrix_power does not support bool tensors", 3)
  end
  local n = math.floor(exponent)
  if n < 0 and not self:is_floating_point() then
    error("matrix_power with a negative exponent requires a " ..
      "floating-point tensor", 3)
  end
  if n == 0 then
    local identity = self:clone():zero_()
    identity:add_(wrap(lib.lt_eye(self:size(-1), self:dtype_id())))
    return identity
  end
  if n == 1 then
    return self:clone()
  end
  local base
  if n < 0 then
    base = self:inverse()
    n = -n
  else
    base = self
  end
  local result = nil
  while n > 0 do
    if n % 2 == 1 then
      result = result == nil and base or result:matmul(base)
    end
    n = math.floor(n / 2)
    if n > 0 then
      base = base:matmul(base)
    end
  end
  return result
end

-- Vandermonde matrix. Tensor:vander treats every row of the last dimension
-- as a vector and appends the N powers as a new last dimension. The default
-- order is increasing (x^0 .. x^(N-1)); torch.vander uses the historical
-- decreasing order instead. Powers are built with ordinary tensor arithmetic,
-- so gradients flow through the input.
function Tensor:vander(N, increasing)
  if self:dim() < 1 then
    error("vander expects a tensor with at least one dimension", 3)
  end
  if self:dtype() == "bool" then
    error("vander does not support bool tensors", 3)
  end
  if N == nil then
    N = self:size(-1)
  else
    N = integer_argument(N, "vander N", 0)
  end
  if increasing == nil then increasing = true end
  if type(increasing) ~= "boolean" then
    error("vander increasing must be a boolean", 3)
  end
  local shape = self:sizes()
  if N == 0 then
    shape[#shape + 1] = 0
    return zeros_tensor(shape, self:dtype_id())
  end
  local powers = {}
  if increasing then
    for power = 0, N - 1 do
      powers[#powers + 1] = self:pow(power)
    end
  else
    for power = N - 1, 0, -1 do
      powers[#powers + 1] = self:pow(power)
    end
  end
  return stack_tensors(powers, self:dim())
end

-- Cross product of two real 3-vectors along `dim`. Inputs keep their rank and
-- broadcast over the other dimensions, matching torch.linalg.cross. The
-- components are ordinary tensor arithmetic, so autograd is preserved.
function Tensor:cross(other, dim)
  if not is_tensor(other) then
    error("cross expects a tensor", 3)
  end
  if self:dtype() == "bool" or other:dtype() == "bool" then
    error("cross does not support bool tensors", 3)
  end
  if self:dtype_id() ~= other:dtype_id() then
    error("cross expects both inputs to have the same dtype", 3)
  end
  local rank = self:dim()
  if rank < 1 then
    error("cross expects tensors with at least one dimension", 3)
  end
  if other:dim() ~= rank then
    error("cross expects inputs with the same number of dimensions", 3)
  end
  dim = dim == nil and -1 or dim
  local axis = dimension_argument(dim, rank, "cross")
  if self:size(axis) ~= 3 or other:size(axis) ~= 3 then
    error(string.format("cross expects dimension %d to have length 3", axis), 3)
  end
  for i = 1, rank do
    if i ~= axis and self:size(i) ~= other:size(i) and
        self:size(i) ~= 1 and other:size(i) ~= 1 then
      error(string.format(
        "cross inputs are not broadcastable at dimension %d", i), 3)
    end
  end
  local a1, a2, a3 = self:select(axis, 1), self:select(axis, 2),
    self:select(axis, 3)
  local b1, b2, b3 = other:select(axis, 1), other:select(axis, 2),
    other:select(axis, 3)
  return stack_tensors({
    a2 * b3 - a3 * b2,
    a3 * b1 - a1 * b3,
    a1 * b2 - a2 * b1,
  }, axis - 1)
end

-- Dot product of two batches of vectors along `dim`, with NumPy-style
-- broadcasting. The reduction dimension is resolved against the broadcast
-- rank, so a lower-rank operand is broadcast before the product is reduced.
function Tensor:vecdot(other, dim)
  if not is_tensor(other) then
    error("vecdot expects a tensor", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float16" and dtype ~= "float32" and dtype ~= "float64" then
    error("vecdot supports float16, float32, and float64 tensors", 3)
  end
  if other:dtype_id() ~= self:dtype_id() then
    error("vecdot expects both inputs to have the same dtype", 3)
  end
  local rank = math.max(self:dim(), other:dim())
  if rank == 0 then
    error("vecdot expects tensors with at least one dimension", 3)
  end
  dim = dim == nil and -1 or dim
  local axis = dimension_argument(dim, rank, "vecdot")
  return (self * other):sum(axis)
end

-- Diagonal over two dimensions. The selected diagonal is appended as the last
-- output dimension, and all other dimensions keep their order, matching
-- torch.diagonal. The implementation permutes the two target dimensions to the
-- end and uses an einsum trace, so it works for batches, offsets, and arbitrary
-- dimension pairs while preserving autograd.
function Tensor:diagonal(offset, dim1, dim2)
  if self:dim() < 2 then
    error("diagonal expects a tensor with at least two dimensions", 3)
  end
  if self:dtype() == "bool" then
    error("diagonal does not support bool tensors", 3)
  end
  offset = offset == nil and 0 or offset
  if type(offset) ~= "number" or offset ~= offset or offset % 1 ~= 0 or
      offset == math.huge or offset == -math.huge then
    error("diagonal offset must be an integer", 3)
  end
  offset = math.floor(offset)
  local rank = self:dim()
  local axis1 = dimension_argument(dim1 == nil and 1 or dim1, rank, "diagonal")
  local axis2 = dimension_argument(dim2 == nil and 2 or dim2, rank, "diagonal")
  if axis1 == axis2 then
    error("diagonal dimensions must be different", 3)
  end

  local sizes = self:sizes()
  local n1, n2 = sizes[axis1], sizes[axis2]
  local start1, start2, length
  if offset >= 0 then
    length = math.max(0, math.min(n1, n2 - offset))
    start1, start2 = 1, offset + 1
  else
    length = math.max(0, math.min(n1 + offset, n2))
    start1, start2 = -offset + 1, 1
  end

  local batch_shape = {}
  for dim = 1, rank do
    if dim ~= axis1 and dim ~= axis2 then
      batch_shape[#batch_shape + 1] = sizes[dim]
    end
  end
  if length == 0 then
    local out_shape = { unpack(batch_shape) }
    out_shape[#out_shape + 1] = 0
    return zeros_tensor(out_shape, self:dtype_id())
  end

  local narrowed = self:narrow(axis1, start1, length)
  narrowed = narrowed:narrow(axis2, start2, length)
  local permutation = {}
  for dim = 1, rank do
    if dim ~= axis1 and dim ~= axis2 then
      permutation[#permutation + 1] = dim
    end
  end
  permutation[#permutation + 1] = axis1
  permutation[#permutation + 1] = axis2
  local permuted = narrowed:permute(unpack(permutation))

  local batch_count = 1
  for i = 1, #batch_shape do
    batch_count = batch_count * batch_shape[i]
  end
  if batch_count == 0 then
    local out_shape = { unpack(batch_shape) }
    out_shape[#out_shape + 1] = length
    return zeros_tensor(out_shape, self:dtype_id())
  end

  local flattened = permuted:reshape(batch_count, length, length)
  local diagonal = einstein_unary(flattened, "...ii->...i")
  local out_shape = { unpack(batch_shape) }
  out_shape[#out_shape + 1] = length
  return diagonal:reshape(out_shape)
end

-- Lower/upper triangular part of a matrix or batch of matrices. Entries on
-- the far side of the requested diagonal are zeroed; the diagonal is 0-based
-- like torch.tril/torch.triu (positive above the main diagonal, negative
-- below). The mask broadcasts over batch dimensions, and masked_fill carries
-- autograd with zero gradient in the zeroed triangle.
local function triangular(tensor, diagonal, keep_lower, name)
  if tensor:dim() < 2 then
    error(name .. " expects a tensor with at least two dimensions", 3)
  end
  diagonal = diagonal_offset(diagonal, name)
  local rows, columns = tensor:size(-2), tensor:size(-1)
  local offsets = offset_grid(rows, columns)
  local zero_mask
  if keep_lower then
    zero_mask = offsets:gt(diagonal)
  else
    zero_mask = offsets:lt(diagonal)
  end
  return tensor:masked_fill(zero_mask, 0)
end

function Tensor:tril(diagonal)
  return triangular(self, diagonal, true, "tril")
end

function Tensor:triu(diagonal)
  return triangular(self, diagonal, false, "triu")
end

-- Sum of the main diagonal of a 2-D matrix.
function Tensor:trace()
  if self:dim() ~= 2 then
    error("trace expects a 2-D matrix", 3)
  end
  return self:diagonal(0, 1, 2):sum()
end

-- torch.diag: extract a diagonal from a 2-D matrix, or build a matrix from a
-- 1-D vector. Construction uses a broadcast mask over a padded column, so the
-- result is differentiable with respect to the vector.
function Tensor:diag(diagonal)
  local rank = self:dim()
  if rank ~= 1 and rank ~= 2 then
    error("diag expects a 1-D or 2-D tensor", 3)
  end
  diagonal = diagonal_offset(diagonal, "diag")
  if rank == 2 then
    return self:diagonal(diagonal, 1, 2)
  end

  local count = self:size(1)
  local size = count + math.abs(diagonal)
  local dtype = self:dtype_id()
  if count == 0 then
    return zeros_tensor({ size, size }, dtype)
  end
  local padding = zeros_tensor({ math.abs(diagonal) }, dtype)
  local padded
  if diagonal >= 0 then
    padded = cat_tensors({ self, padding }, 0)
  else
    padded = cat_tensors({ padding, self }, 0)
  end
  local columns = padded:reshape(size, 1):expand(size, size)
  local offsets = offset_grid(size, size)
  local keep = offsets:eq(diagonal)
  return columns:masked_fill(keep:logical_not(), 0)
end

-- Embed the last dimension of a tensor on a diagonal of two new trailing
-- dimensions. Inputs of shape (*, n) produce (*, size, size) with
-- size = n + |offset|; leading dimensions are treated as batch dimensions.
-- The construction uses a padded column broadcast against a diagonal mask, so
-- it stays differentiable with respect to the input.
function Tensor:diag_embed(diagonal)
  if self:dim() < 1 then
    error("diag_embed expects a tensor with at least one dimension", 3)
  end
  diagonal = diagonal_offset(diagonal, "diag_embed")
  local count = self:size(-1)
  local size = count + math.abs(diagonal)
  local dtype = self:dtype_id()
  local batch_shape = {}
  for i = 1, self:dim() - 1 do
    batch_shape[i] = self:size(i)
  end
  local out_shape = {}
  for i = 1, #batch_shape do out_shape[i] = batch_shape[i] end
  out_shape[#out_shape + 1] = size
  out_shape[#out_shape + 1] = size
  if count == 0 then
    return zeros_tensor(out_shape, dtype)
  end
  local padding_shape = {}
  for i = 1, #batch_shape do padding_shape[i] = batch_shape[i] end
  padding_shape[#padding_shape + 1] = math.abs(diagonal)
  local padding = zeros_tensor(padding_shape, dtype)
  local padded
  if diagonal >= 0 then
    padded = cat_tensors({ self, padding }, self:dim() - 1)
  else
    padded = cat_tensors({ padding, self }, self:dim() - 1)
  end
  local columns = padded:unsqueeze(-1):expand(unpack(out_shape))
  local keep = offset_grid(size, size):eq(diagonal)
  return columns:masked_fill(keep:logical_not(), 0)
end

function Tensor:mv(vector)
  return wrap(lib.lt_mv(self.h, vector.h))
end

function Tensor:outer(vector)
  return wrap(lib.lt_outer(self.h, vector.h))
end

-- Kronecker product. Shapes are aligned from the right; missing leading
-- dimensions count as 1 and each output dimension is the product of the two
-- corresponding input dimensions.
function Tensor:kron(other)
  if not is_tensor(other) then
    error("kron expects another tensor", 3)
  end
  local a_shape, b_shape = {}, {}
  for i = 1, self:dim() do a_shape[i] = self:size(i) end
  for i = 1, other:dim() do b_shape[i] = other:size(i) end
  local rank = math.max(#a_shape, #b_shape)
  if rank == 0 then
    return self:mul(other)
  end
  local a, b = {}, {}
  for i = 1, rank - #a_shape do a[i] = 1 end
  for i = 1, #a_shape do a[#a + 1] = a_shape[i] end
  for i = 1, rank - #b_shape do b[i] = 1 end
  for i = 1, #b_shape do b[#b + 1] = b_shape[i] end

  local out_shape, a_reshape, b_reshape = {}, {}, {}
  local expanded = {}
  for i = 1, rank do
    out_shape[i] = a[i] * b[i]
    a_reshape[#a_reshape + 1] = a[i]
    a_reshape[#a_reshape + 1] = 1
    b_reshape[#b_reshape + 1] = 1
    b_reshape[#b_reshape + 1] = b[i]
    expanded[#expanded + 1] = a[i]
    expanded[#expanded + 1] = b[i]
  end
  local left = self:reshape(a_reshape):expand(unpack(expanded))
  local right = other:reshape(b_reshape):expand(unpack(expanded))
  return left:mul(right):reshape(out_shape)
end

function Tensor:addmm(mat1, mat2, beta, alpha)
  return wrap(lib.lt_addmm(self.h, mat1.h, mat2.h,
    beta == nil and 1 or beta, alpha == nil and 1 or alpha))
end

function Tensor:baddbmm(mat1, mat2, beta, alpha)
  return wrap(lib.lt_baddbmm(self.h, mat1.h, mat2.h,
    beta == nil and 1 or beta, alpha == nil and 1 or alpha))
end

local function softmax_any_dim(self, dim, log)
  dim = dim or self:dim()
  local cdim = self:_cdim(dim)
  local sizes = self:sizes()
  local dtype = self:dtype_id()
  local last = #sizes - 1
  if #sizes >= 1 and cdim == last and (dtype == 0 or dtype == 1) then
    local pointer = log and lib.lt_fused_log_softmax(self.h) or
      lib.lt_fused_softmax(self.h)
    if pointer ~= nil then return wrap(pointer) end
    return wrap(log and lib.lt_log_softmax(self.h, cdim) or
      lib.lt_softmax(self.h, cdim))
  end
  if #sizes >= 2 and cdim ~= last and (dtype == 0 or dtype == 1) then
    -- The fused kernels require the softmax dimension to be contiguous.
    -- Move it last, run the fused kernel, then move the view back.
    local moved = self:transpose(cdim + 1, last + 1):contiguous()
    local result = log and moved:log_softmax(last + 1) or
      moved:softmax(last + 1)
    return result:transpose(cdim + 1, last + 1)
  end
  return wrap(log and lib.lt_log_softmax(self.h, cdim) or
    lib.lt_softmax(self.h, cdim))
end

function Tensor:softmax(dim)
  return softmax_any_dim(self, dim, false)
end

function Tensor:log_softmax(dim)
  return softmax_any_dim(self, dim, true)
end

-- ---------------------------------------------------------------------------
-- shape manipulation
-- ---------------------------------------------------------------------------

local function shape_args(...)
  local args = { ... }
  if #args == 1 and type(args[1]) == "table" then
    args = args[1]
  end
  return args
end

function Tensor:reshape(...)
  local shape = shape_args(...)
  local array, ndim = shape_to_cdata(shape)
  return wrap(lib.lt_reshape(self.h, ndim, array))
end

function Tensor:view(...)
  local shape = shape_args(...)
  local array, ndim = shape_to_cdata(shape)
  return wrap(lib.lt_view(self.h, ndim, array))
end

function Tensor:transpose(dim0, dim1)
  return wrap(lib.lt_transpose(self.h, self:_cdim(dim0), self:_cdim(dim1)))
end

function Tensor:t()
  return wrap(lib.lt_t(self.h))
end

function Tensor:permute(...)
  local dims = shape_args(...)
  local n = #dims
  if n ~= self:dim() then
    error(string.format("permute expects %d dimensions, got %d", self:dim(), n), 3)
  end
  local array = ffi.new("int64_t[?]", n)
  for i = 1, n do
    array[i - 1] = self:_cdim(dims[i])
  end
  return wrap(lib.lt_permute(self.h, n, array))
end

function Tensor:swapaxes(dim0, dim1)
  return self:transpose(dim0, dim1)
end

local function axis_list(value, name)
  if type(value) == "number" then
    return { value }
  end
  if type(value) == "table" and #value > 0 then
    return value
  end
  error(name .. " must be an integer or a non-empty table", 3)
end

-- Move one or more axes to new positions, preserving the order of the other
-- axes. Multiple moves are applied in sequence, matching NumPy/PyTorch.
function Tensor:moveaxis(source, destination)
  local sources = axis_list(source, "moveaxis source")
  local destinations = axis_list(destination, "moveaxis destination")
  if #sources ~= #destinations then
    error("moveaxis source and destination must have the same length", 3)
  end
  local result = self
  for i = 1, #sources do
    local rank = result:dim()
    local from = dimension_argument(sources[i], rank, "moveaxis")
    local to = dimension_argument(destinations[i], rank, "moveaxis")
    local order = {}
    for d = 1, rank do
      if d ~= from then order[#order + 1] = d end
    end
    table.insert(order, to, from)
    result = result:permute(unpack(order))
  end
  return result
end

function Tensor:squeeze(dim)
  if dim == nil then
    return wrap(lib.lt_squeeze_all(self.h))
  end
  return wrap(lib.lt_squeeze(self.h, self:_cdim(dim)))
end

function Tensor:unsqueeze(dim)
  return wrap(lib.lt_unsqueeze(self.h, self:_cdim(dim)))
end

function Tensor:flatten(start_dim, end_dim)
  local s = start_dim or 1
  local e = end_dim or self:dim()
  return wrap(lib.lt_flatten(self.h, self:_cdim(s), self:_cdim(e)))
end

function Tensor:expand(...)
  local shape = shape_args(...)
  local array, ndim = shape_to_cdata(shape)
  return wrap(lib.lt_expand(self.h, ndim, array))
end

-- Split along one dimension into views. A number gives fixed-size chunks
-- (the last may be shorter); a table gives explicit section sizes.
function Tensor:split(split_size_or_sections, dim)
  local axis = dim == nil and 1 or dim
  local length = self:size(axis)
  local parts = {}
  if type(split_size_or_sections) == "table" then
    local total = 0
    for i = 1, #split_size_or_sections do
      local size = split_size_or_sections[i]
      if type(size) ~= "number" or size ~= size or size % 1 ~= 0 or
          size < 0 or size == math.huge then
        error("split sections must be non-negative integers", 3)
      end
      total = total + size
    end
    if total ~= length then
      error("split sections must sum to the dimension size", 3)
    end
    local start = 1
    for i = 1, #split_size_or_sections do
      local size = split_size_or_sections[i]
      parts[i] = self:narrow(axis, start, size)
      start = start + size
    end
    return parts
  end
  if type(split_size_or_sections) ~= "number" or
      split_size_or_sections ~= split_size_or_sections or
      split_size_or_sections % 1 ~= 0 or split_size_or_sections <= 0 or
      split_size_or_sections == math.huge then
    error("split size must be a positive integer", 3)
  end
  local size = math.floor(split_size_or_sections)
  local start = 1
  while start <= length do
    local part_size = math.min(size, length - start + 1)
    parts[#parts + 1] = self:narrow(axis, start, part_size)
    start = start + part_size
  end
  return parts
end

-- Split into at most `chunks` roughly equal parts along one dimension.
function Tensor:chunk(chunks, dim)
  if type(chunks) ~= "number" or chunks ~= chunks or chunks % 1 ~= 0 or
      chunks <= 0 or chunks == math.huge then
    error("chunk count must be a positive integer", 3)
  end
  local axis = dim == nil and 1 or dim
  local length = self:size(axis)
  local size = math.ceil(length / math.floor(chunks))
  if size < 1 then size = 1 end
  return self:split(size, axis)
end

-- Remove one dimension and return its slices as a Lua array.
function Tensor:unbind(dim)
  local axis = dim == nil and 1 or dim
  local length = self:size(axis)
  local parts = {}
  for i = 1, length do
    parts[i] = self:select(axis, i)
  end
  return parts
end

-- Circular shift. Without dims the tensor is flattened, rolled, and reshaped;
-- with dims, shifts and dims may be integers or equally long lists.
function Tensor:roll(shifts, dims)
  local function normalize(value, name)
    if type(value) == "number" then
      return { value }
    end
    if type(value) ~= "table" or #value == 0 then
      error(name .. " must be an integer or a non-empty table", 3)
    end
    return value
  end
  local shift_list = normalize(shifts, "roll shifts")
  for i = 1, #shift_list do
    local value = shift_list[i]
    if type(value) ~= "number" or value ~= value or value % 1 ~= 0 or
        value == math.huge or value == -math.huge then
      error("roll shifts must contain integers", 3)
    end
  end
  if dims == nil then
    if #shift_list ~= 1 then
      error("roll without dims expects one shift", 3)
    end
    local flat = self:reshape(-1)
    local length = flat:size(1)
    if length == 0 then
      return flat:reshape(self:sizes())
    end
    local shift = shift_list[1] % length
    if shift == 0 then
      return flat:reshape(self:sizes())
    end
    local result = cat_tensors({
      flat:narrow(1, length - shift + 1, shift),
      flat:narrow(1, 1, length - shift),
    }, 0)
    return result:reshape(self:sizes())
  end

  local dim_list = normalize(dims, "roll dims")
  for i = 1, #dim_list do
    local value = dim_list[i]
    if type(value) ~= "number" or value ~= value or value % 1 ~= 0 or
        value == math.huge or value == -math.huge then
      error("roll dims must contain integers", 3)
    end
  end
  if #shift_list ~= #dim_list then
    error("roll shifts and dims must have the same length", 3)
  end
  local result = self
  for i = 1, #shift_list do
    local dim = dim_list[i]
    local length = result:size(dim)
    if length > 0 then
      local shift = shift_list[i] % length
      if shift ~= 0 then
        local tail = result:narrow(dim, length - shift + 1, shift)
        local head = result:narrow(dim, 1, length - shift)
        result = cat_tensors({ tail, head }, result:_cdim(dim))
      end
    end
  end
  return result
end

-- Tile a tensor by a table of repeats. Missing leading repeats default to 1
-- and extra repeats promote the input with leading size-1 dimensions.
function Tensor:tile(reps)
  if type(reps) ~= "table" or #reps == 0 then
    error("tile expects a non-empty table of repeats", 3)
  end
  for i = 1, #reps do
    local value = reps[i]
    if type(value) ~= "number" or value ~= value or value % 1 ~= 0 or
        value < 0 or value == math.huge then
      error("tile repeats must be non-negative integers", 3)
    end
  end
  local result = self
  local rank = result:dim()
  if #reps > rank then
    local promoted = {}
    for _ = 1, #reps - rank do promoted[#promoted + 1] = 1 end
    for i = 1, rank do promoted[#promoted + 1] = result:size(i) end
    result = result:reshape(promoted)
    rank = #reps
  elseif #reps < rank then
    local padded = {}
    for _ = 1, rank - #reps do padded[#padded + 1] = 1 end
    for i = 1, #reps do padded[#padded + 1] = reps[i] end
    reps = padded
  end

  for i = 1, #reps do
    local repeats = reps[i]
    if repeats ~= 1 then
      local sizes = result:sizes()
      local length = sizes[i]
      local expanded = {}
      for d = 1, #sizes do
        if d == i then expanded[#expanded + 1] = repeats end
        expanded[#expanded + 1] = sizes[d]
      end
      result = result:unsqueeze(i):expand(unpack(expanded))
      local merged = {}
      for d = 1, #sizes do
        if d == i then
          merged[#merged + 1] = repeats * length
        else
          merged[#merged + 1] = sizes[d]
        end
      end
      result = result:reshape(merged)
    end
  end
  return result
end

-- Repeat each element (dim given) or each flattened element (dim nil).
function Tensor:repeat_interleave(repeats, dim)
  local counts = {}
  local length
  if dim == nil then
    length = self:numel()
  else
    length = self:size(dim)
  end
  if type(repeats) == "number" then
    if repeats ~= repeats or repeats % 1 ~= 0 or repeats < 0 or
        repeats == math.huge then
      error("repeat_interleave repeats must be a non-negative integer", 3)
    end
    for i = 1, length do counts[i] = repeats end
  elseif is_tensor(repeats) then
    local dtype = repeats:dtype_id()
    if dtype ~= 2 and dtype ~= 3 and dtype ~= 4 and dtype ~= 5 then
      error("repeat_interleave repeats must be an integer tensor", 3)
    end
    if repeats:dim() ~= 1 or repeats:size(1) ~= length then
      error("repeat_interleave repeats must be a 1-D tensor matching the " ..
        "selected dimension", 3)
    end
    counts = repeats:to("int64"):totable()
  else
    error("repeat_interleave repeats must be a number or an integer tensor", 3)
  end

  local indices = {}
  for i = 1, length do
    local count = counts[i]
    if type(count) ~= "number" or count ~= count or count % 1 ~= 0 or
        count < 0 or count == math.huge then
      error("repeat_interleave repeats must be non-negative integers", 3)
    end
    for _ = 1, count do indices[#indices + 1] = i end
  end
  local index_tensor = Tensor.from_table(indices, { dtype = 2 })
  if dim == nil then
    return self:reshape(-1):index_select(1, index_tensor)
  end
  return self:index_select(dim, index_tensor)
end

-- ---------------------------------------------------------------------------
-- reductions
-- ---------------------------------------------------------------------------

local function reduction(self, fn, dim, keepdim)
  return wrap(fn(self.h, self:_cdim(dim), keepdim and 1 or 0))
end

function Tensor:sum(dim, keepdim)
  return reduction(self, lib.lt_sum, dim, keepdim)
end

function Tensor:mean(dim, keepdim)
  return reduction(self, lib.lt_mean, dim, keepdim)
end

function Tensor:prod(dim, keepdim)
  return reduction(self, lib.lt_prod, dim, keepdim)
end

function Tensor:max(dim, keepdim)
  local values = reduction(self, lib.lt_max, dim, keepdim)
  if dim == nil then
    return values
  end
  return values, reduction(self, lib.lt_argmax, dim, keepdim):add(1)
end

function Tensor:min(dim, keepdim)
  local values = reduction(self, lib.lt_min, dim, keepdim)
  if dim == nil then
    return values
  end
  return values, reduction(self, lib.lt_argmin, dim, keepdim):add(1)
end

local function extrema_dimensions(tensor, dim, name)
  if type(dim) == "number" then
    return { dimension_argument(dim, tensor:dim(), name) }
  end
  if type(dim) ~= "table" then
    error(name .. " dim must be an integer or a table of integers", 3)
  end
  if #dim == 0 then
    error(name .. " dim table must not be empty", 3)
  end
  local dims, seen = {}, {}
  for i = 1, #dim do
    local resolved = dimension_argument(dim[i], tensor:dim(), name)
    if seen[resolved] then
      error(name .. " dim table must not repeat dimensions", 3)
    end
    seen[resolved] = true
    dims[#dims + 1] = resolved
  end
  table.sort(dims)
  return dims
end

-- Minimum/maximum over one dimension or a table of dimensions, returning
-- values only. Dimensions are reduced from the highest index down so the
-- remaining indices stay valid when keepdim is false.
local function extrema_dim(tensor, dim, keepdim, is_min, name)
  if dim == nil then
    if is_min then return tensor:min() end
    return tensor:max()
  end
  local dims = extrema_dimensions(tensor, dim, name)
  local values = tensor
  for i = #dims, 1, -1 do
    if is_min then
      values = values:min(dims[i], keepdim)
    else
      values = values:max(dims[i], keepdim)
    end
  end
  return values
end

function Tensor:amin(dim, keepdim)
  return extrema_dim(self, dim, keepdim, true, "amin")
end

function Tensor:amax(dim, keepdim)
  return extrema_dim(self, dim, keepdim, false, "amax")
end

-- Min and max values in one call, without materializing argmin/argmax. When
-- `dim` is nil the result is two 0-dim tensors; otherwise both results have
-- the reduced shape. `keepdim` is ignored without a dimension, matching
-- PyTorch's aminmax.
function Tensor:aminmax(dim, keepdim)
  if dim == nil then
    return self:min(), self:max()
  end
  if type(dim) ~= "number" or dim % 1 ~= 0 then
    error("aminmax dim must be an integer", 3)
  end
  if keepdim ~= nil and type(keepdim) ~= "boolean" then
    error("aminmax keepdim must be a boolean", 3)
  end
  return self:min(dim, keepdim), self:max(dim, keepdim)
end

-- Count non-zero entries. With no dimension this is a 0-dim int64 tensor;
-- with an integer or a table of dimensions the reduced dimensions are
-- removed, matching torch.count_nonzero.
function Tensor:count_nonzero(dim)
  local counts = self:ne(0):to("int64")
  if dim == nil then
    return counts:sum()
  end
  if type(dim) == "number" then
    if dim % 1 ~= 0 then
      error("count_nonzero dim must be an integer", 3)
    end
    local rank = self:dim()
    if dim == 0 or dim > rank or dim < -rank then
      error("count_nonzero dim " .. tostring(dim) .. " is out of range", 3)
    end
    return counts:sum(dim)
  end
  if type(dim) ~= "table" then
    error("count_nonzero dim must be an integer or a table of integers", 3)
  end
  if #dim == 0 then
    error("count_nonzero dim table must not be empty", 3)
  end
  local rank = self:dim()
  local dims, seen = {}, {}
  for i = 1, #dim do
    local value = dim[i]
    if type(value) ~= "number" or value % 1 ~= 0 then
      error("count_nonzero dim must contain integers", 3)
    end
    local resolved = value
    if resolved < 0 then resolved = resolved + rank + 1 end
    if resolved < 1 or resolved > rank then
      error("count_nonzero dim " .. tostring(value) .. " is out of range", 3)
    end
    if seen[resolved] then
      error("count_nonzero dim table must not repeat dimensions", 3)
    end
    seen[resolved] = true
    dims[#dims + 1] = resolved
  end
  table.sort(dims)
  -- Reduce from the highest dimension down so earlier indices stay valid.
  for i = #dims, 1, -1 do
    counts = counts:sum(dims[i])
  end
  return counts
end

-- Sum ignoring NaNs. Non-finite infinities are kept, and an all-NaN slice
-- sums to zero. NaN entries contribute zero gradient through the where.
function Tensor:nansum(dim, keepdim)
  local mask = self:isnan()
  local zeros = full_like_value(self, 0)
  local replaced = wrap(lib.lt_where(mask.h, zeros.h, self.h))
  return replaced:sum(dim, keepdim)
end

-- Mean ignoring NaNs. The denominator is the number of non-NaN entries in
-- each slice, so an all-NaN slice returns NaN. Only floating-point inputs are
-- accepted, matching PyTorch's nanmean.
function Tensor:nanmean(dim, keepdim)
  local dtype = self:dtype()
  if dtype ~= "float16" and dtype ~= "float32" and dtype ~= "float64" then
    error("nanmean supports float16, float32, and float64 tensors", 3)
  end
  local mask = self:isnan()
  local zeros = full_like_value(self, 0)
  local replaced = wrap(lib.lt_where(mask.h, zeros.h, self.h))
  local numerator = replaced:sum(dim, keepdim)
  local counts = mask:logical_not():to("int64"):sum(dim, keepdim)
  return numerator:div(counts:to(dtype))
end

-- Variance ignoring NaNs. The mean and squared deviations use only non-NaN
-- entries; the denominator is the valid count minus the unbiased correction,
-- and slices with too few valid entries return NaN.
function Tensor:nanvar(dim, unbiased, keepdim)
  if unbiased == nil then unbiased = true end
  if type(unbiased) ~= "boolean" then
    error("nanvar unbiased must be a boolean", 3)
  end
  local dtype = self:dtype()
  if dtype ~= "float16" and dtype ~= "float32" and dtype ~= "float64" then
    error("nanvar supports float16, float32, and float64 tensors", 3)
  end
  local mask = self:isnan()
  local zeros = full_like_value(self, 0)
  local replaced = wrap(lib.lt_where(mask.h, zeros.h, self.h))
  local counts = mask:logical_not():to(dtype):sum(dim, keepdim)
  local mean = replaced:sum(dim, keepdim):div(counts)
  local centered = replaced:sub(mean)
  local squared = centered:mul(centered)
  squared = wrap(lib.lt_where(mask.h, zeros.h, squared.h))
  local total = squared:sum(dim, keepdim)
  local denominator = counts
  if unbiased then
    denominator = counts:sub(1)
  end
  local invalid
  if unbiased then
    invalid = counts:le(1)
  else
    invalid = counts:eq(0)
  end
  denominator = wrap(lib.lt_where(invalid.h,
    full_like_value(counts, 0).h, denominator.h))
  return total:div(denominator)
end

function Tensor:nanstd(dim, unbiased, keepdim)
  return self:nanvar(dim, unbiased, keepdim):sqrt()
end

-- ATen returns 0-based indices; Lua users expect 1-based ones, so shift.
function Tensor:argmax(dim, keepdim)
  return reduction(self, lib.lt_argmax, dim, keepdim):add(1)
end

function Tensor:argmin(dim, keepdim)
  return reduction(self, lib.lt_argmin, dim, keepdim):add(1)
end

function Tensor:var(dim, unbiased, keepdim)
  if unbiased == nil then unbiased = true end
  return wrap(lib.lt_var(self.h, self:_cdim(dim), keepdim and 1 or 0, unbiased and 1 or 0))
end

function Tensor:std(dim, unbiased, keepdim)
  if unbiased == nil then unbiased = true end
  return wrap(lib.lt_std(self.h, self:_cdim(dim), keepdim and 1 or 0, unbiased and 1 or 0))
end

function Tensor:norm(p, dim, keepdim)
  if p == nil then p = 2 end
  if type(p) ~= "number" then
    error("norm: p must be a number", 3)
  end
  return wrap(lib.lt_norm(self.h, p, self:_cdim(dim), keepdim and 1 or 0))
end

function Tensor:cumsum(dim)
  return wrap(lib.lt_cumsum(self.h, self:_cdim(dim or 1)))
end

function Tensor:cumprod(dim)
  return wrap(lib.lt_cumprod(self.h, self:_cdim(dim or 1)))
end

-- Cumulative log-sum-exp along one dimension, built from stable logaddexp
-- steps and carrying autograd through the concatenated slices.
function Tensor:logcumsumexp(dim)
  local axis = dim == nil and self:dim() or dim
  local length = self:size(axis)
  if length == 0 then
    return self:clone()
  end
  local parts = {}
  local running = self:narrow(axis, 1, 1)
  parts[1] = running
  for i = 2, length do
    running = running:logaddexp(self:narrow(axis, i, 1))
    parts[i] = running
  end
  return cat_tensors(parts, self:_cdim(axis))
end

function Tensor:all(dim, keepdim)
  return reduction(self, lib.lt_all, dim, keepdim)
end

function Tensor:any(dim, keepdim)
  return reduction(self, lib.lt_any, dim, keepdim)
end

function Tensor:logsumexp(dim, keepdim)
  return wrap(lib.lt_logsumexp(self.h, self:_cdim(dim), keepdim and 1 or 0))
end

local function sizes_equal(a, b)
  if #a ~= #b then
    return false
  end
  for i = 1, #a do
    if a[i] ~= b[i] then
      return false
    end
  end
  return true
end

-- Trapezoidal rule along one dimension. `x` holds the sample positions: it
-- may be a 1-D tensor with one value per sample or a tensor shaped like the
-- input, in which case spacing may vary per slice. With no `x`, `dx` is the
-- uniform spacing. The integrated dimension is removed from the result.
function Tensor:trapezoid(x, opts)
  if opts == nil then
    opts = {}
  elseif type(opts) ~= "table" then
    error("trapezoid options must be a table", 3)
  end
  if x ~= nil and not is_tensor(x) then
    error("trapezoid x must be a Tensor or nil", 3)
  end
  if self:dim() < 1 then
    error("trapezoid expects a tensor with at least one dimension", 3)
  end
  if not self:is_floating_point() then
    error("trapezoid expects a floating-point tensor; cast integer or bool " ..
      "inputs first", 3)
  end
  local dim = opts.dim
  if dim == nil then
    dim = self:dim()
  elseif type(dim) ~= "number" or dim % 1 ~= 0 or dim == 0 then
    error("trapezoid dim must be a non-zero integer", 3)
  end
  if dim < 0 then
    dim = self:dim() + dim + 1
  end
  if dim < 1 or dim > self:dim() then
    error("trapezoid dim is out of range", 3)
  end
  local dx = opts.dx
  if dx ~= nil and type(dx) ~= "number" then
    error("trapezoid dx must be a number", 3)
  end
  if x ~= nil and dx ~= nil then
    error("trapezoid accepts either x or dx, not both", 3)
  end

  local n = self:size(dim)
  if x ~= nil then
    if x:dtype() == "bool" then
      error("trapezoid x does not support bool tensors", 3)
    end
    if not x:is_floating_point() then
      x = x:to(self:dtype_id())
    elseif x:dtype_id() ~= self:dtype_id() then
      error("trapezoid x dtype must match the input dtype", 3)
    end
    if x:dim() == 1 then
      if x:size(1) ~= n then
        error("trapezoid x must have one value per sample along dim", 3)
      end
    elseif x:dim() ~= self:dim() or not sizes_equal(x:sizes(), self:sizes()) then
      error("trapezoid x must be 1-D with size(dim) values or have the " ..
        "same shape as the input", 3)
    end
  end

  if n <= 1 then
    -- No intervals: a zero with the integrated dimension removed.
    return self:narrow(dim, 1, 0):sum(dim)
  end

  local values = self:narrow(dim, 1, n - 1)
    :add(self:narrow(dim, 2, n - 1)):mul(0.5)
  local delta
  if x == nil then
    delta = dx == nil and 1 or dx
  elseif x:dim() == 1 then
    delta = x:narrow(1, 2, n - 1):sub(x:narrow(1, 1, n - 1))
    local shape = {}
    for i = 1, self:dim() do shape[i] = 1 end
    shape[dim] = n - 1
    delta = delta:reshape(shape)
  else
    delta = x:narrow(dim, 2, n - 1):sub(x:narrow(dim, 1, n - 1))
  end
  return values:mul(delta):sum(dim)
end

-- ---------------------------------------------------------------------------
-- comparisons / helpers
-- ---------------------------------------------------------------------------

function Tensor:equal(other)
  if not is_tensor(other) then
    return false
  end
  local a, b = self:sizes(), other:sizes()
  if #a ~= #b then
    return false
  end
  for i = 1, #a do
    if a[i] ~= b[i] then
      return false
    end
  end
  return self:eq(other):all():item() ~= false
end

-- Elementwise closeness with NumPy/PyTorch semantics: values are close when
-- they are exactly equal or when |a - b| <= atol + rtol * |b|. Exact equality
-- keeps matching infinities close, which the tolerance formula cannot express
-- because inf - inf is NaN. `equal_nan` additionally treats matching NaNs as
-- close. The result is a detached bool tensor.
function Tensor:isclose(other, rtol, atol, equal_nan)
  if not is_tensor(other) then
    error("isclose expects a tensor", 3)
  end
  if self:dtype() == "bool" or other:dtype() == "bool" then
    error("isclose does not support bool tensors", 3)
  end
  rtol = rtol == nil and 1e-5 or rtol
  atol = atol == nil and 1e-8 or atol
  if type(rtol) ~= "number" or rtol < 0 then
    error("isclose rtol must be a non-negative number", 3)
  end
  if type(atol) ~= "number" or atol < 0 then
    error("isclose atol must be a non-negative number", 3)
  end
  if equal_nan ~= nil and type(equal_nan) ~= "boolean" then
    error("isclose equal_nan must be a boolean", 3)
  end
  local difference = self:sub(other):abs()
  local tolerance = other:abs():mul(rtol):add(atol)
  local finite = self:isfinite():logical_and(other:isfinite())
  local close = self:eq(other):logical_or(
    finite:logical_and(difference:le(tolerance)))
  if equal_nan then
    close = close:logical_or(self:isnan():logical_and(other:isnan()))
  end
  return close
end

function Tensor:allclose(other, rtol, atol, equal_nan)
  return self:isclose(other, rtol, atol, equal_nan):all():item() ~= false
end

-- ---------------------------------------------------------------------------
-- conversion to Lua values
-- ---------------------------------------------------------------------------

local function half_to_float(bits)
  local sign = (bits >= 32768) and -1 or 1
  local exponent = math.floor(bits / 1024) % 32
  local mantissa = bits % 1024
  if exponent == 0 then
    if mantissa == 0 then
      return sign * 0.0
    end
    return sign * mantissa * 2 ^ -24
  elseif exponent == 31 then
    if mantissa == 0 then
      return sign * math.huge
    end
    return 0 / 0
  end
  return sign * (1 + mantissa / 1024) * 2 ^ (exponent - 15)
end

local function make_reader(dtype, pointer)
  if dtype == 0 then
    local p = ffi.cast("float*", pointer)
    return function(i) return canonical_double(p[i]) end
  elseif dtype == 1 then
    local p = ffi.cast("double*", pointer)
    return function(i) return canonical_double(p[i]) end
  elseif dtype == 2 then
    local p = ffi.cast("int64_t*", pointer)
    return function(i) return tonumber(p[i]) end
  elseif dtype == 3 then
    local p = ffi.cast("int32_t*", pointer)
    return function(i) return p[i] end
  elseif dtype == 4 then
    local p = ffi.cast("int16_t*", pointer)
    return function(i) return p[i] end
  elseif dtype == 5 then
    local p = ffi.cast("int8_t*", pointer)
    return function(i) return p[i] end
  elseif dtype == 6 then
    local p = ffi.cast("uint8_t*", pointer)
    return function(i) return p[i] end
  elseif dtype == 7 then
    local p = ffi.cast("uint8_t*", pointer)
    return function(i) return p[i] ~= 0 end
  elseif dtype == 8 then
    local p = ffi.cast("uint16_t*", pointer)
    return function(i) return half_to_float(p[i]) end
  end
  error("unsupported dtype id " .. tostring(dtype), 3)
end

function Tensor:totable()
  local contiguous = self:contiguous()
  local count = contiguous:numel()
  if count == 0 then return {} end
  local pointer = lib.lt_data_ptr(contiguous.h)
  if pointer == nil then
    error(last_error(), 3)
  end
  local reader = make_reader(contiguous:dtype_id(), pointer)
  local values = {}
  for i = 1, count do
    values[i] = reader(i - 1)
  end
  local sizes = contiguous:sizes()
  if #sizes == 0 then
    return values[1]
  end
  local offset = 1
  local function build(dimension)
    local size = sizes[dimension]
    local result = {}
    if dimension == #sizes then
      for i = 1, size do
        result[i] = values[offset]
        offset = offset + 1
      end
    else
      for i = 1, size do
        result[i] = build(dimension + 1)
      end
    end
    return result
  end
  return build(1)
end

function Tensor:iter()
  local contiguous = self:contiguous()
  local count = contiguous:numel()
  local pointer = lib.lt_data_ptr(contiguous.h)
  if pointer == nil then
    error(last_error(), 3)
  end
  local reader = make_reader(contiguous:dtype_id(), pointer)
  local index = 0
  local function next_value()
    if index >= count then
      return nil
    end
    local value = reader(index)
    index = index + 1
    return value
  end
  -- returning the tensor as the loop state keeps it alive during for-in
  return next_value, contiguous, nil
end

-- ---------------------------------------------------------------------------
-- functional utilities (luafun replaces Moses here)
-- ---------------------------------------------------------------------------

function Tensor:fun()
  local fun = require("fun")
  return fun.iter(self:totable())
end

function Tensor:map(fn)
  if type(fn) ~= "function" then
    error("map expects a function", 3)
  end
  local values = {}
  local index = 0
  for value in self:iter() do
    index = index + 1
    values[index] = fn(value)
  end
  return from_flat(values, self:sizes(), { dtype = self:dtype_id() })
end

function Tensor:filter(predicate)
  if type(predicate) ~= "function" then
    error("filter expects a function", 3)
  end
  local values = {}
  for value in self:iter() do
    if predicate(value) then
      values[#values + 1] = value
    end
  end
  return from_flat(values, { #values }, { dtype = self:dtype_id() })
end

function Tensor:reduce(fn, initial)
  if type(fn) ~= "function" then
    error("reduce expects a function", 3)
  end
  local accumulator = initial
  local started = initial ~= nil
  for value in self:iter() do
    if not started then
      accumulator = value
      started = true
    else
      accumulator = fn(accumulator, value)
    end
  end
  return accumulator
end

function Tensor:apply(fn)
  if type(fn) ~= "function" then
    error("apply expects a function", 3)
  end
  local mapped = self:map(fn)
  self:copy_(mapped)
  return self
end

-- ---------------------------------------------------------------------------
-- serialization
-- ---------------------------------------------------------------------------

function Tensor:save(path)
  if type(path) ~= "string" then
    error("save expects a file path", 3)
  end
  check(lib.lt_save(self.h, path))
  return self
end

-- ---------------------------------------------------------------------------
-- printing
-- ---------------------------------------------------------------------------

local function format_float(value)
  if value ~= value then
    return "nan"
  end
  if value == math.huge then
    return "inf"
  end
  if value == -math.huge then
    return "-inf"
  end
  if value == 0 then
    return "0."
  end
  local magnitude = math.abs(value)
  if magnitude < 1e-4 or magnitude >= 1e4 then
    return string.format("%.4e", value)
  end
  local text = string.format("%.4f", value)
  text = text:gsub("0+$", "")
  if text:sub(-1) == "." then
    return text
  end
  return text
end

local function render(value, level, format_scalar)
  if type(value) ~= "table" then
    return format_scalar(value)
  end
  if #value == 0 then
    return "[]"
  end
  local leaf = type(value[1]) ~= "table"
  local parts = {}
  for i = 1, #value do
    parts[i] = render(value[i], level + 1, format_scalar)
  end
  if leaf then
    return "[" .. table.concat(parts, ", ") .. "]"
  end
  local indent = string.rep(" ", 8 + level * 2)
  return "[" .. table.concat(parts, ",\n" .. indent) .. "]"
end

function Tensor:format()
  local dtype = self:dtype()
  local suffix = ""
  if dtype ~= "float32" and dtype ~= "int64" and dtype ~= "bool" then
    suffix = ", dtype=torch." .. dtype
  end
  if self:requires_grad() then
    suffix = suffix .. ", requires_grad=True"
  end
  local count = self:numel()
  if count == 0 then
    return "tensor([]" .. suffix .. ")"
  end
  local is_float = is_dtype_floating(self:dtype_id())
  local function format_scalar(value)
    if type(value) == "boolean" then
      return tostring(value)
    end
    if is_float then
      return format_float(value)
    end
    return string.format("%d", math.floor(value))
  end
  if count > 1000 then
    local dims = self:sizes()
    local function values(t)
      local parts = {}
      for _, value in ipairs(t:totable()) do
        parts[#parts + 1] = format_scalar(value)
      end
      return parts
    end
    if #dims == 1 then
      local n = dims[1]
      local head = math.min(4, n)
      local tail = math.min(4, n)
      local parts = values(self:narrow(1, 1, head))
      if head + tail < n then parts[#parts + 1] = "..." end
      if n > head then
        for _, value in ipairs(values(self:narrow(1, n - tail + 1, tail))) do
          parts[#parts + 1] = value
        end
      end
      return string.format("tensor([%s], shape=[%s]%s)",
        table.concat(parts, ", "), table.concat(dims, ", "), suffix)
    elseif #dims == 2 then
      local rows, columns = dims[1], dims[2]
      local head_rows = math.min(3, rows)
      local tail_rows = math.min(3, rows)
      local head_cols = math.min(3, columns)
      local tail_cols = math.min(3, columns)
      local function row(t)
        local parts = values(t:narrow(1, 1, head_cols))
        if head_cols + tail_cols < columns then parts[#parts + 1] = "..." end
        if columns > head_cols then
          for _, value in ipairs(values(t:narrow(1,
            columns - tail_cols + 1, tail_cols))) do
            parts[#parts + 1] = value
          end
        end
        return "[" .. table.concat(parts, ", ") .. "]"
      end
      local lines = {}
      for r = 1, head_rows do lines[#lines + 1] = row(self:select(1, r)) end
      if head_rows + tail_rows < rows then lines[#lines + 1] = "..." end
      if rows > head_rows then
        for r = rows - tail_rows + 1, rows do
          lines[#lines + 1] = row(self:select(1, r))
        end
      end
      local body = "[\n        " .. table.concat(lines, ",\n        ") ..
        "]"
      return string.format("tensor(%s, shape=[%s]%s)", body,
        table.concat(dims, ", "), suffix)
    else
      return string.format("tensor(shape=[%s], dtype=torch.%s, numel=%d%s)",
        table.concat(dims, ", "), dtype, count, suffix)
    end
  end
  local body = render(self:totable(), 0, format_scalar)
  return "tensor(" .. body .. suffix .. ")"
end

function Tensor:print()
  print(self:format())
  return self
end

-- ---------------------------------------------------------------------------
-- metamethods
-- ---------------------------------------------------------------------------

mt.__index = function(self, key)
  if type(key) == "number" then
    if self:dim() == 0 then
      error("cannot index a 0-dim tensor; use item()", 2)
    end
    return Tensor.select(self, 1, key)
  end
  if is_tensor(key) then
    if key:dtype() == "bool" then
      return Tensor.masked_select(self, key)
    end
    error("only boolean tensors can be used as index keys", 2)
  end
  local value = Tensor[key]
  if value ~= nil then
    if profiler.is_enabled() and type(value) == "function" then
      local wrapper = profiled[key]
      if wrapper == nil then
        wrapper = function(...) return profiler.call(key, value, ...) end
        profiled[key] = wrapper
      end
      return wrapper
    end
    return value
  end
  error(string.format("torch.Tensor has no field '%s'", tostring(key)), 2)
end

mt.__newindex = function(self, key, value)
  if type(key) ~= "number" then
    rawset(self, key, value)
    return
  end
  local dims = self:dim()
  if dims == 0 then
    error("cannot index a 0-dim tensor; use fill()", 2)
  end
  if dims == 1 then
    if type(value) == "number" or type(value) == "boolean" then
      self:set(key, value)
      return
    end
    error("1-d assignment expects a number or boolean", 2)
  end
  if not is_tensor(value) then
    error("cannot assign a non-tensor into a slice", 2)
  end
  self:select(1, key):copy_(value)
end

mt.__add = function(a, b)
  if is_tensor(a) then return a:add(b) end
  return b:add(a)
end
mt.__sub = function(a, b)
  if is_tensor(a) then return a:sub(b) end
  return b:neg():add(a)
end
mt.__mul = function(a, b)
  if is_tensor(a) then return a:mul(b) end
  return b:mul(a)
end
mt.__div = function(a, b)
  if is_tensor(a) then return a:div(b) end
  return b:reciprocal():mul(a)
end
mt.__pow = function(a, b)
  if is_tensor(a) then return a:pow(b) end
  if a <= 0 then
    error("number ^ Tensor requires a positive base", 2)
  end
  return b:mul(math.log(a)):exp()
end
mt.__unm = function(a) return a:neg() end
mt.__tostring = function(a) return a:format() end
mt.__len = function(a)
  if a:dim() == 0 then
    return 1
  end
  return a:size(1)
end

-- Tensor(...) is a Torch7-style float32 constructor.
setmetatable(Tensor, {
  __call = function(_, ...)
    return tensor_ctor(...)
  end,
})

Tensor._wrap = wrap
Tensor._from_flat = from_flat
Tensor._create = create
Tensor._parse_shape_opts = parse_shape_opts
Tensor._is_options = is_options
Tensor.from_table = from_table
Tensor.from_scalar = from_scalar
Tensor.is_tensor = is_tensor
Tensor.resolve_dtype = resolve_dtype
Tensor.ALL_DIMS = ALL_DIMS
Tensor.dtype_names = dtype_names
Tensor._mt = mt

return Tensor
