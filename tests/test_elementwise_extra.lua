-- Signed-infinity predicates, NaN-aware reductions, addcmul/addcdiv, and lerp.

local torch = require("torch")

test("isposinf and isneginf distinguish signed infinities", function()
  local values = torch.tensor({math.huge, -math.huge, 0 / 0, 0, 2},
    {dtype = torch.float64})
  assert_eq(values:isposinf():totable(),
    {true, false, false, false, false})
  assert_eq(values:isneginf():totable(),
    {false, true, false, false, false})
  assert_eq(torch.isposinf(values):totable(), values:isposinf():totable())
  assert_eq(torch.isneginf(values):totable(), values:isneginf():totable())

  local half = torch.tensor({math.huge, -math.huge},
    {dtype = torch.float16})
  assert_eq(half:isposinf():totable(), {true, false})
  assert_eq(half:isneginf():totable(), {false, true})

  local integers = torch.tensor({1, 2}, {dtype = torch.int64})
  assert_eq(integers:isposinf():totable(), {false, false})
  assert_eq(integers:isneginf():totable(), {false, false})

  local graded = torch.tensor({math.huge}, {requires_grad = true})
  assert_false(graded:isposinf():requires_grad())
  assert_error(function() torch.isposinf(1) end, "expects a tensor")
  assert_error(function() torch.isneginf(1) end, "expects a tensor")
end)

test("nansum ignores NaNs across the whole tensor and dimensions", function()
  local values = torch.tensor({{0 / 0, 2}, {3, 0 / 0}},
    {dtype = torch.float64})
  assert_eq(values:nansum():item(), 5)
  assert_eq(values:nansum(1):totable(), {3, 2})
  assert_eq(values:nansum(2):totable(), {2, 3})
  assert_shape(values:nansum(1, true), {1, 2})
  assert_eq(torch.nansum(values):item(), 5)

  local all_nan = torch.tensor({0 / 0, 0 / 0}, {dtype = torch.float64})
  assert_eq(all_nan:nansum():item(), 0)

  local integers = torch.tensor({1, 2, 3}, {dtype = torch.int64})
  assert_eq(integers:nansum():item(), 6)
  assert_eq(integers:nansum():dtype(), "int64")
end)

test("nansum passes gradients only through non-NaN entries", function()
  local values = torch.tensor({1, 0 / 0, 3},
    {dtype = torch.float64, requires_grad = true})
  values:nansum():backward()
  assert_eq(values:grad():totable(), {1, 0, 1})
end)

test("nanmean ignores NaNs and returns NaN for empty slices", function()
  local values = torch.tensor({{0 / 0, 2}, {3, 4}},
    {dtype = torch.float64})
  assert_eq(values:nanmean():item(), 3)
  assert_eq(values:nanmean(1):totable(), {3, 3})
  assert_eq(values:nanmean(2):totable(), {2, 3.5})
  assert_shape(values:nanmean(2, true), {2, 1})
  assert_eq(torch.nanmean(values):item(), 3)

  local all_nan = torch.tensor({0 / 0, 0 / 0}, {dtype = torch.float64})
  assert_true(all_nan:nanmean():item() ~= all_nan:nanmean():item())

  local half = torch.tensor({0 / 0, 2}, {dtype = torch.float16})
  assert_eq(half:nanmean():dtype(), "float16")
  assert_eq(half:nanmean():item(), 2)
end)

test("nanmean is differentiable and rejects non-floating inputs", function()
  local values = torch.tensor({1, 0 / 0, 3},
    {dtype = torch.float64, requires_grad = true})
  values:nanmean():backward()
  assert_eq(values:grad():totable(), {0.5, 0, 0.5})

  assert_error(function()
    torch.tensor({1, 2}, {dtype = torch.int64}):nanmean()
  end, "float")
  assert_error(function()
    torch.tensor({true, false}):nanmean()
  end, "float")
end)

test("addcmul multiplies two tensors and scales the result", function()
  local input = torch.tensor({1, 2, 3}, {dtype = torch.float64})
  local first = torch.tensor({4, 5, 6}, {dtype = torch.float64})
  local second = torch.tensor({2, 2, 2}, {dtype = torch.float64})
  assert_eq(input:addcmul(first, second):totable(), {9, 12, 15})
  assert_eq(input:addcmul(first, second, 0.5):totable(), {5, 7, 9})
  assert_eq(torch.addcmul(input, first, second, 2):totable(), {17, 22, 27})
  -- The second operand broadcasts.
  assert_eq(input:addcmul(first, torch.tensor({2},
    {dtype = torch.float64})):totable(), {9, 12, 15})
  assert_error(function() input:addcmul(first) end, "two tensors")
  assert_error(function() input:addcmul(first, second, "x") end, "number")
  assert_error(function() torch.addcmul(1, first, second) end,
    "expects a tensor")
end)

test("addcdiv divides two tensors and scales the result", function()
  local input = torch.tensor({1, 2, 3}, {dtype = torch.float64})
  local first = torch.tensor({4, 5, 6}, {dtype = torch.float64})
  local second = torch.tensor({2, 2, 2}, {dtype = torch.float64})
  assert_eq(input:addcdiv(first, second):totable(), {3, 4.5, 6})
  assert_eq(input:addcdiv(first, second, 2):totable(), {5, 7, 9})
  assert_eq(torch.addcdiv(input, first, second):totable(), {3, 4.5, 6})
  assert_error(function() torch.addcdiv(1, first, second) end,
    "expects a tensor")
end)

test("addcmul and addcdiv are differentiable through all operands", function()
  local input = torch.tensor({1, 2}, {dtype = torch.float64,
    requires_grad = true})
  local first = torch.tensor({3, 4}, {dtype = torch.float64,
    requires_grad = true})
  local second = torch.tensor({2, 5}, {dtype = torch.float64,
    requires_grad = true})
  input:addcmul(first, second):sum():backward()
  assert_eq(input:grad():totable(), {1, 1})
  assert_eq(first:grad():totable(), {2, 5})
  assert_eq(second:grad():totable(), {3, 4})

  local dividend = torch.tensor({4, 6}, {dtype = torch.float64,
    requires_grad = true})
  local divisor = torch.tensor({2, 3}, {dtype = torch.float64,
    requires_grad = true})
  local base = torch.tensor({0, 0}, {dtype = torch.float64,
    requires_grad = true})
  base:addcdiv(dividend, divisor):sum():backward()
  assert_true(base:grad():allclose(torch.ones(2,
    {dtype = torch.float64})))
  assert_true(dividend:grad():allclose(torch.tensor({0.5, 1 / 3},
    {dtype = torch.float64}), 1e-12, 1e-12))
end)

test("lerp interpolates with scalar and tensor weights", function()
  local input = torch.tensor({1, 2, 3}, {dtype = torch.float64})
  local finish = torch.tensor({4, 5, 6}, {dtype = torch.float64})
  assert_eq(input:lerp(finish, 0.5):totable(), {2.5, 3.5, 4.5})
  assert_eq(input:lerp(finish, 0):totable(), {1, 2, 3})
  assert_eq(input:lerp(finish, 1):totable(), {4, 5, 6})
  assert_eq(torch.lerp(input, finish, 0.25):totable(),
    {1.75, 2.75, 3.75})
  assert_eq(input:lerp(finish, torch.tensor({0.25},
    {dtype = torch.float64})):totable(), {1.75, 2.75, 3.75})
end)

test("lerp passes gradients to both endpoints and a tensor weight", function()
  local input = torch.tensor({1, 2, 3},
    {dtype = torch.float64, requires_grad = true})
  local finish = torch.tensor({4, 5, 6},
    {dtype = torch.float64, requires_grad = true})
  local weight = torch.tensor({0.25},
    {dtype = torch.float64, requires_grad = true})
  input:lerp(finish, weight):sum():backward()
  assert_true(input:grad():allclose(torch.full({3}, 0.75,
    {dtype = torch.float64})))
  assert_true(finish:grad():allclose(torch.full({3}, 0.25,
    {dtype = torch.float64})))
  assert_eq(weight:grad():item(), 9)
end)

test("lerp validates dtypes, weights, and aliases", function()
  local input = torch.tensor({1, 2}, {dtype = torch.float64})
  local finish = torch.tensor({3, 4}, {dtype = torch.float64})
  assert_error(function()
    torch.tensor({1, 2}, {dtype = torch.int64}):lerp(
      torch.tensor({3, 4}, {dtype = torch.int64}), 0.5)
  end, "float")
  assert_error(function()
    input:lerp(torch.tensor({3, 4}, {dtype = torch.float32}), 0.5)
  end, "dtype")
  assert_error(function()
    input:lerp(finish, torch.tensor({0.5}, {dtype = torch.float32}))
  end, "weight")
  assert_error(function() input:lerp(finish) end, "weight")
  assert_error(function() input:lerp(finish, "x") end, "weight")
  assert_error(function() input:lerp(1, 0.5) end, "end tensor")
  assert_error(function() torch.lerp(1, finish, 0.5) end,
    "expects a tensor")
end)
