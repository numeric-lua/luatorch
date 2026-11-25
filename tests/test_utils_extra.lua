-- Tensor utility gaps: broadcast_shapes/broadcast_tensors, take, logit,
-- and clamp_min/clamp_max.

local torch = require("torch")

test("broadcast_shapes aligns shapes from the right", function()
  assert_eq(torch.broadcast_shapes({3, 1, 5}, {2, 5}), {3, 2, 5})
  assert_eq(torch.broadcast_shapes({}, {3}), {3})
  assert_eq(torch.broadcast_shapes({2, 3}), {2, 3})
  assert_eq(torch.broadcast_shapes({{2, 1}, {1, 3}}), {2, 3})
  assert_eq(torch.broadcast_shapes({0, 1}, {1, 4}), {0, 4})
  assert_eq(torch.broadcast_shapes(), {})
end)

test("broadcast_shapes rejects invalid and incompatible shapes", function()
  assert_error(function() torch.broadcast_shapes({2}, {3}) end,
    "incompatible")
  assert_error(function() torch.broadcast_shapes({2, -1}) end,
    "non-negative")
  assert_error(function() torch.broadcast_shapes({2.5}) end,
    "integer")
  assert_error(function() torch.broadcast_shapes(2) end, "shape tables")
end)

test("broadcast_tensors expands every input to the common shape", function()
  local column = torch.tensor({{1}, {2}}, {dtype = torch.float64})
  local row = torch.tensor({3, 4}, {dtype = torch.float64})
  local a, b = torch.broadcast_tensors(column, row)
  assert_shape(a, {2, 2})
  assert_shape(b, {2, 2})
  assert_eq(a:totable(), {{1, 1}, {2, 2}})
  assert_eq(b:totable(), {{3, 4}, {3, 4}})

  local scalar, matrix = torch.broadcast_tensors(
    torch.tensor(5, {dtype = torch.float64}),
    torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64}))
  assert_shape(scalar, {2, 2})
  assert_eq(scalar:totable(), {{5, 5}, {5, 5}})
  assert_eq(matrix:totable(), {{1, 2}, {3, 4}})

  local list_a, list_b = torch.broadcast_tensors({column, row})
  assert_eq(list_a:totable(), a:totable())
  assert_eq(list_b:totable(), b:totable())
end)

test("broadcast_tensors handles scalar pairs and validates inputs", function()
  local a, b = torch.broadcast_tensors(torch.tensor(5), torch.tensor(7))
  assert_eq(a:sizes(), {})
  assert_eq(b:sizes(), {})
  assert_eq(a:item(), 5)
  assert_error(function() torch.broadcast_tensors() end, "at least one")
  assert_error(function() torch.broadcast_tensors(1) end, "tensors")
end)

test("take gathers flattened elements with 1-based indices", function()
  local matrix = torch.tensor({{1, 2, 3}, {4, 5, 6}},
    {dtype = torch.float64})
  assert_eq(matrix:take(torch.tensor({6, 1, 4},
    {dtype = torch.int64})):totable(), {6, 1, 4})
  assert_eq(matrix:take(torch.tensor({{1, 2}, {6, 5}},
    {dtype = torch.int64})):totable(), {{1, 2}, {6, 5}})
  assert_eq(matrix:take(torch.tensor(3, {dtype = torch.int64})):item(), 3)
  assert_shape(matrix:take(torch.tensor({}, {dtype = torch.int64})), {0})
  assert_eq(torch.take(matrix, torch.tensor({2, 4},
    {dtype = torch.int64})):totable(), {2, 4})
end)

test("take is differentiable and validates index tensors", function()
  local matrix = torch.tensor({{1, 2}, {3, 4}},
    {dtype = torch.float64, requires_grad = true})
  matrix:take(torch.tensor({1, 3}, {dtype = torch.int64})):sum():backward()
  assert_true(matrix:grad():allclose(torch.tensor({{1, 0}, {1, 0}},
    {dtype = torch.float64})))

  assert_error(function()
    matrix:take(torch.tensor({1.0}, {dtype = torch.float64}))
  end, "int8/int16/int32/int64")
  assert_error(function()
    matrix:take(torch.tensor({true}))
  end, "int8/int16/int32/int64")
  assert_error(function()
    matrix:take(torch.tensor({0}, {dtype = torch.int64}))
  end, "index")
end)

test("logit is the inverse sigmoid and clamps with eps", function()
  local values = torch.tensor({0.25, 0.5, 0.75}, {dtype = torch.float64})
  local result = values:logit()
  assert_near(result:get(1), math.log(1 / 3), 1e-12)
  assert_eq(result:get(2), 0)
  assert_near(result:get(3), math.log(3), 1e-12)
  assert_eq(torch.logit(values):get(2), 0)

  local edges = torch.tensor({0, 0.5, 1}, {dtype = torch.float64})
  local clamped = edges:logit(1e-6)
  assert_near(clamped:get(1), math.log(1e-6 / (1 - 1e-6)), 1e-9)
  assert_eq(clamped:get(2), 0)
  assert_near(clamped:get(3), -math.log(1e-6 / (1 - 1e-6)), 1e-9)
end)

test("logit is differentiable and validates its arguments", function()
  local values = torch.tensor({0.25},
    {dtype = torch.float64, requires_grad = true})
  values:logit():backward()
  assert_near(values:grad():get(1), 1 / (0.25 * 0.75), 1e-12)

  assert_error(function() torch.tensor({1, 2}, {dtype = torch.int64}):logit()
  end, "float")
  assert_error(function() torch.tensor({0.5}):logit(-0.1) end, "eps")
  assert_error(function() torch.tensor({0.5}):logit(0.5) end, "eps")
  assert_error(function() torch.tensor({0.5}):logit("0") end, "eps")
  assert_error(function() torch.logit(1) end, "expects a tensor")
end)

test("clamp_min and clamp_max replace only out-of-range values", function()
  local values = torch.tensor({-1, 0, 2, 0 / 0}, {dtype = torch.float64})
  local low = values:clamp_min(0)
  assert_eq(low:narrow(1, 1, 3):totable(), {0, 0, 2})
  assert_true(low:get(4) ~= low:get(4), "clamp_min should preserve NaN")

  local high = values:clamp_max(0)
  assert_eq(high:narrow(1, 1, 3):totable(), {-1, 0, 0})
  assert_true(high:get(4) ~= high:get(4), "clamp_max should preserve NaN")

  assert_eq(torch.clamp_min(values, 1):narrow(1, 1, 3):totable(), {1, 1, 2})
  assert_eq(torch.clamp_max(values, 1):narrow(1, 1, 3):totable(), {-1, 0, 1})
  local infinite = values:clamp_min(math.huge)
  assert_eq(infinite:narrow(1, 1, 3):totable(),
    {math.huge, math.huge, math.huge})
  assert_true(infinite:get(4) ~= infinite:get(4))
end)

test("clamp_min/clamp_max support in-place calls and autograd", function()
  local values = torch.tensor({-1, 0, 2}, {dtype = torch.float64})
  local result = values:clamp_min_(0)
  assert_true(result == values)
  assert_eq(values:totable(), {0, 0, 2})

  local maximum = torch.tensor({-1, 0, 2}, {dtype = torch.float64})
  maximum:clamp_max_(0)
  assert_eq(maximum:totable(), {-1, 0, 0})

  local low = torch.tensor({-1, 2},
    {dtype = torch.float64, requires_grad = true})
  low:clamp_min(0):sum():backward()
  assert_eq(low:grad():totable(), {0, 1})

  local high = torch.tensor({-1, 2},
    {dtype = torch.float64, requires_grad = true})
  high:clamp_max(0):sum():backward()
  assert_eq(high:grad():totable(), {1, 0})
end)

test("clamp_min/clamp_max reject integer/bool tensors and bad bounds", function()
  assert_error(function()
    torch.tensor({1, 2}, {dtype = torch.int64}):clamp_min(0)
  end, "float")
  assert_error(function() torch.tensor({true}):clamp_max(0) end, "float")
  assert_error(function() torch.tensor({1}):clamp_min("0") end, "bound")
  assert_error(function() torch.tensor({1}):clamp_max(0 / 0) end, "bound")
  assert_error(function() torch.clamp_min(1, 0) end, "expects a tensor")
end)
