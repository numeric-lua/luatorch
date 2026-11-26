-- NaN-aware extrema, kthvalue, unique_consecutive, and logcumsumexp.

local torch = require("torch")

test("fmax and fmin ignore NaNs and broadcast", function()
  local nan = 0 / 0
  local a = torch.tensor({1, nan, 3}, {dtype = torch.float64})
  local b = torch.tensor({2, 4, nan}, {dtype = torch.float64})
  assert_eq(a:fmax(b):totable(), {2, 4, 3})
  assert_eq(a:fmin(b):totable(), {1, 4, 3})
  assert_eq(torch.fmax(a, b):totable(), {2, 4, 3})
  assert_eq(torch.fmin(a, b):totable(), {1, 4, 3})

  local both = torch.tensor({nan}, {dtype = torch.float64})
  assert_true(both:fmax(both):get(1) ~= both:fmax(both):get(1))
  assert_true(both:fmin(both):get(1) ~= both:fmin(both):get(1))

  local rows = torch.tensor({{1, 3}, {2, 4}}, {dtype = torch.float64})
  local column = torch.tensor({2}, {dtype = torch.float64})
  assert_eq(rows:fmax(column):totable(), {{2, 3}, {2, 4}})
  assert_eq(torch.tensor({1, 2}, {dtype = torch.int64}):fmax(
    torch.tensor({2, 1}, {dtype = torch.int64})):totable(), {2, 2})
end)

test("fmax and fmin are differentiable and validate operands", function()
  local left = torch.tensor({3, 1},
    {dtype = torch.float64, requires_grad = true})
  left:fmax(torch.tensor({2, 4}, {dtype = torch.float64})):sum():backward()
  assert_eq(left:grad():totable(), {1, 0})

  local lower = torch.tensor({3, 1},
    {dtype = torch.float64, requires_grad = true})
  lower:fmin(torch.tensor({2, 4}, {dtype = torch.float64})):sum():backward()
  assert_eq(lower:grad():totable(), {0, 1})

  local values = torch.tensor({1, 2})
  assert_error(function() values:fmax(1) end, "tensor")
  assert_error(function()
    values:fmax(torch.tensor({1, 2}, {dtype = torch.float64}))
  end, "dtype")
  assert_error(function() torch.tensor({true}):fmin(torch.tensor({false}))
  end, "bool")
  assert_error(function() torch.fmax(1, values) end, "first argument")
end)

test("kthvalue returns values and 1-based indices", function()
  local matrix = torch.tensor({{3, 1, 2}, {6, 5, 4}},
    {dtype = torch.float64})
  local values, indices = matrix:kthvalue(2)
  assert_eq(values:totable(), {2, 5})
  assert_eq(indices:totable(), {3, 2})
  assert_eq(indices:dtype(), "int64")

  local kept_values, kept_indices = matrix:kthvalue(2, 1, true)
  assert_shape(kept_values, {1, 3})
  assert_eq(kept_values:totable(), {{6, 5, 4}})
  assert_eq(kept_indices:totable(), {{2, 2, 2}})
  assert_eq(torch.kthvalue(matrix, 1, 2):totable(), {1, 4})
end)

test("kthvalue is differentiable and validates k", function()
  local values = torch.tensor({3, 1, 2},
    {dtype = torch.float64, requires_grad = true})
  values:kthvalue(1):sum():backward()
  assert_eq(values:grad():totable(), {0, 1, 0})

  local matrix = torch.tensor({{1, 2}, {3, 4}})
  assert_error(function() matrix:kthvalue(0) end, "positive")
  assert_error(function() matrix:kthvalue(3) end, "out of range")
  assert_error(function() matrix:kthvalue(1.5) end, "integer")
  assert_error(function() torch.kthvalue(1, 1) end, "expects a tensor")
end)

test("unique_consecutive collapses only adjacent duplicates", function()
  local values = torch.tensor({1, 1, 2, 2, 2, 3, 1},
    {dtype = torch.float64})
  assert_eq(values:unique_consecutive():totable(), {1, 2, 3, 1})

  local unique, inverse, counts = values:unique_consecutive(true, true)
  assert_eq(unique:totable(), {1, 2, 3, 1})
  assert_eq(inverse:totable(), {1, 1, 2, 2, 2, 3, 4})
  assert_eq(counts:totable(), {2, 3, 1, 1})
  assert_eq(inverse:dtype(), "int64")
  assert_eq(counts:dtype(), "int64")
  assert_eq(torch.unique_consecutive(values):totable(),
    {1, 2, 3, 1})
end)

test("unique_consecutive handles empty and bool inputs", function()
  local empty = torch.tensor({}, {dtype = torch.float64})
  local values, inverse, counts = empty:unique_consecutive(true, true)
  assert_shape(values, {0})
  assert_shape(inverse, {0})
  assert_shape(counts, {0})

  local flags = torch.tensor({true, true, false, true})
  assert_eq(flags:unique_consecutive():totable(), {true, false, true})
  assert_error(function() torch.unique_consecutive(1) end,
    "expects a tensor")
end)

test("logcumsumexp computes stable cumulative values", function()
  local values = torch.tensor({0, 0, 0}, {dtype = torch.float64})
  local result = values:logcumsumexp()
  assert_near(result:get(1), 0, 1e-12)
  assert_near(result:get(2), math.log(2), 1e-12)
  assert_near(result:get(3), math.log(3), 1e-12)
  assert_eq(torch.logcumsumexp(values):get(3), result:get(3))

  local matrix = torch.tensor({{1, 2, 3}, {4, 5, 6}},
    {dtype = torch.float64})
  assert_near(matrix:logcumsumexp(2):get(1, 2), math.log(math.exp(1) +
    math.exp(2)), 1e-12)
  assert_near(matrix:logcumsumexp(1):get(2, 1),
    math.log(math.exp(1) + math.exp(4)), 1e-12)
end)

test("logcumsumexp is differentiable and handles empty dimensions", function()
  local values = torch.tensor({0, 0, 0},
    {dtype = torch.float64, requires_grad = true})
  values:logcumsumexp():sum():backward()
  assert_true(values:grad():allclose(
    torch.tensor({11 / 6, 5 / 6, 1 / 3}, {dtype = torch.float64}),
    1e-12, 1e-12))

  local empty = torch.tensor({}, {dtype = torch.float64})
  assert_shape(empty:logcumsumexp(), {0})
  assert_error(function() torch.tensor({1, 2}, {dtype = torch.int64})
    :logcumsumexp() end, "float")
  assert_error(function() torch.logcumsumexp(1) end, "expects a tensor")
end)
