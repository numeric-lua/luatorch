-- Creation helpers, dtype promotion, and the like-constructor regression.

local torch = require("torch")

test("ones_like and full_like preserve shape, dtype, and values", function()
  local input = torch.zeros(2, 3)
  local ones = torch.ones_like(input)
  assert_shape(ones, {2, 3})
  assert_eq(ones:dtype(), "float32")
  assert_eq(ones:sum():item(), 6)

  local doubles = torch.ones_like(input, { dtype = torch.float64 })
  assert_eq(doubles:dtype(), "float64")

  local filled = torch.full_like(input, 7)
  assert_eq(filled:totable(), { {7, 7, 7}, {7, 7, 7} })
  local float_filled = torch.full_like(input, 2.5, { dtype = torch.float64 })
  assert_eq(float_filled:dtype(), "float64")
  assert_eq(float_filled:get(1, 1), 2.5)
end)

test("bincount and histc remain callable after the helper repair", function()
  assert_eq(torch.bincount(torch.tensor({1, 1, 2})):totable(), {0, 2, 1})
  assert_eq(torch.bincount(torch.tensor({1, 1, 2}),
    torch.tensor({1, 2, 3}, { dtype = torch.float64 })):totable(),
    {0, 3, 3})
  assert_eq(torch.histc(torch.tensor({1, 2, 3}), 2):totable(), {1, 2})
end)

test("empty_like matches shape and supports dtype overrides", function()
  local input = torch.zeros(2, 3)
  local empty = torch.empty_like(input)
  assert_shape(empty, {2, 3})
  assert_eq(empty:dtype(), "float32")

  local doubles = torch.empty_like(input, { dtype = torch.float64 })
  assert_shape(doubles, {2, 3})
  assert_eq(doubles:dtype(), "float64")
  assert_error(function() torch.empty_like(1) end, "expects a tensor")
end)

test("logspace generates geometric sequences", function()
  assert_tensor(torch.logspace(0, 2, 3), {1, 10, 100}, 1e-4)
  assert_tensor(torch.logspace(0, 2, 3, 2), {1, 2, 4}, 1e-4)
  assert_tensor(torch.logspace(0, 2, 3, { dtype = torch.float64 }),
    {1, 10, 100}, 1e-12)
  assert_eq(torch.logspace(0, 2, 3, { dtype = torch.float64 }):dtype(),
    "float64")
  assert_error(function() torch.logspace(0, 2, 3, 0) end, "positive")
  assert_error(function() torch.logspace(0, 2, 3, { dtype = torch.int64 }) end,
    "floating")
end)

test("randint samples the requested shape and dtype", function()
  local values = torch.randint(0, 10, {4, 5})
  assert_shape(values, {4, 5})
  assert_eq(values:dtype(), "int64")
  for _, value in ipairs(values:flatten():totable()) do
    assert_true(value >= 0 and value < 10)
    assert_eq(value % 1, 0)
  end

  local upper_only = torch.randint(10, {3})
  assert_shape(upper_only, {3})
  local floats = torch.randint(0, 10, 2, 3, { dtype = torch.float32 })
  assert_shape(floats, {2, 3})
  assert_eq(floats:dtype(), "float32")

  assert_error(function() torch.randint(0, 0, {2}) end, "high > low")
  assert_error(function() torch.randint(0.5, 2, {2}) end, "integers")
  assert_error(function() torch.randint(0, 10, {-1}) end,
    "non-negative integers")
end)

test("promote_types follows integer, float, and bool categories", function()
  assert_eq(torch.promote_types(torch.int32, torch.int32), torch.int32)
  assert_eq(torch.promote_types(torch.bool, torch.int16), torch.int16)
  assert_eq(torch.promote_types(torch.int64, torch.float32), torch.float32)
  assert_eq(torch.promote_types(torch.float16, torch.float64), torch.float64)
  assert_eq(torch.promote_types(torch.uint8, torch.int8), torch.int16)
  assert_eq(torch.promote_types("uint8", "int16"), torch.int16)
  assert_eq(torch.promote_types("int8", "int64"), torch.int64)
  assert_error(function() torch.promote_types("nope", torch.int32) end,
    "unknown dtype")
end)

test("result_type promotes two tensor dtypes", function()
  local integers = torch.zeros(2, { dtype = torch.int32 })
  local doubles = torch.zeros(2, { dtype = torch.float64 })
  assert_eq(torch.result_type(integers, doubles), torch.float64)
  assert_eq(torch.result_type(doubles, doubles), torch.float64)
  assert_error(function() torch.result_type(integers, 1) end,
    "expects two tensors")
end)
