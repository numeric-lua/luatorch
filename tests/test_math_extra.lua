-- amin/amax, NaN-aware variance, logaddexp/logaddexp2, hypot, and frac.

local torch = require("torch")

test("amin and amax reduce values only", function()
  local a = torch.tensor({{1, 5, 3}, {4, 2, 6}}, {dtype = torch.float64})
  assert_eq(a:amin():item(), 1)
  assert_eq(a:amax():item(), 6)
  assert_eq(a:amin(1):totable(), {1, 2, 3})
  assert_eq(a:amax(1):totable(), {4, 5, 6})
  assert_eq(a:amin(2):totable(), {1, 2})
  assert_eq(a:amax(2):totable(), {5, 6})
  assert_shape(a:amin(2, true), {2, 1})
  assert_shape(a:amax(2, true), {2, 1})
  assert_eq(torch.amin(a):item(), 1)
  assert_eq(torch.amax(a):item(), 6)
end)

test("amin and amax accept dimension tables and negative dimensions", function()
  local a = torch.tensor({{1, 5, 3}, {4, 2, 6}}, {dtype = torch.float64})
  assert_eq(a:amin({1, 2}):item(), 1)
  assert_eq(a:amax({1, 2}):item(), 6)
  assert_eq(a:amin({2, 1}):item(), 1)
  assert_eq(a:amax({2, 1}):item(), 6)
  assert_shape(a:amin({1, 2}, true), {1, 1})
  assert_eq(a:amax(-1):totable(), {5, 6})
end)

test("amin and amax validate dimensions and tensor arguments", function()
  local a = torch.tensor({{1, 2}, {3, 4}})
  assert_error(function() a:amin(0) end, "dimension")
  assert_error(function() a:amax(3) end, "dimension")
  assert_error(function() a:amin({}) end, "empty")
  assert_error(function() a:amin({1, 1}) end, "repeat")
  assert_error(function() a:amax({1.5}) end, "integer")
  assert_error(function() torch.amin(1) end, "expects a tensor")
  assert_error(function() torch.amax(1) end, "expects a tensor")
end)

test("nanvar and nanstd ignore NaNs with biased and unbiased denominators", function()
  local values = torch.tensor({1, 0 / 0, 3}, {dtype = torch.float64})
  assert_eq(values:nanvar():item(), 2)
  assert_eq(values:nanvar(nil, false):item(), 1)
  assert_near(values:nanstd():item(), math.sqrt(2), 1e-12)
  assert_near(values:nanstd(nil, false):item(), 1, 1e-12)
  assert_eq(torch.nanvar(values):item(), 2)
  assert_near(torch.nanstd(values):item(), math.sqrt(2), 1e-12)

  local matrix = torch.tensor({{1, 0 / 0}, {3, 5}},
    {dtype = torch.float64})
  assert_eq(matrix:nanvar(1):totable(), {2, 0 / 0})
  assert_shape(matrix:nanvar(1, true, true), {1, 2})
end)

test("nanvar handles empty slices and rejects non-floating inputs", function()
  local all_nan = torch.tensor({0 / 0, 0 / 0}, {dtype = torch.float64})
  assert_true(all_nan:nanvar():item() ~= all_nan:nanvar():item())
  assert_true(all_nan:nanstd():item() ~= all_nan:nanstd():item())

  local single = torch.tensor({0 / 0, 2}, {dtype = torch.float64})
  assert_true(single:nanvar():item() ~= single:nanvar():item())
  assert_eq(single:nanvar(nil, false):item(), 0)

  assert_error(function()
    torch.tensor({1, 2}, {dtype = torch.int64}):nanvar()
  end, "float")
  assert_error(function()
    torch.tensor({1, 2}):nanvar(nil, 1)
  end, "boolean")
end)

test("nanvar and nanstd are differentiable only through valid entries", function()
  local values = torch.tensor({1, 0 / 0, 3},
    {dtype = torch.float64, requires_grad = true})
  values:nanvar():backward()
  assert_true(values:grad():allclose(torch.tensor({-2, 0, 2},
    {dtype = torch.float64}), 1e-12, 1e-12))

  local std_values = torch.tensor({1, 0 / 0, 3},
    {dtype = torch.float64, requires_grad = true})
  std_values:nanstd():backward()
  assert_true(std_values:grad():allclose(
    torch.tensor({-math.sqrt(2) / 2, 0, math.sqrt(2) / 2},
      {dtype = torch.float64}), 1e-12, 1e-12))
end)

test("logaddexp and logaddexp2 are stable and broadcast", function()
  local x = torch.tensor({1, 2}, {dtype = torch.float64})
  local y = torch.tensor({2, 1}, {dtype = torch.float64})
  local expected = math.log(math.exp(1) + math.exp(2))
  assert_near(x:logaddexp(y):get(1), expected, 1e-12)
  assert_near(x:logaddexp(y):get(2), expected, 1e-12)
  assert_near(x:logaddexp2(y):get(1), math.log(6) / math.log(2), 1e-12)

  local big = torch.tensor({1000}, {dtype = torch.float64})
  local small = torch.tensor({-1000}, {dtype = torch.float64})
  assert_eq(big:logaddexp(small):item(), 1000)
  assert_eq(torch.tensor({0}, {dtype = torch.float64})
    :logaddexp(torch.tensor({0}, {dtype = torch.float64})):get(1),
    math.log(2))

  local rows = torch.tensor({{1, 2}}, {dtype = torch.float64})
  local column = torch.tensor({{3}, {4}}, {dtype = torch.float64})
  assert_shape(rows:logaddexp(column), {2, 2})
  assert_near(torch.logaddexp(rows, column):get(1, 1),
    math.log(math.exp(1) + math.exp(3)), 1e-12)
end)

test("logaddexp is differentiable and validates operands", function()
  local x = torch.tensor({1, 2},
    {dtype = torch.float64, requires_grad = true})
  local y = torch.tensor({3, 4},
    {dtype = torch.float64, requires_grad = true})
  x:logaddexp(y):sum():backward()
  assert_true(x:grad():allclose(torch.tensor({0.1192, 0.1192},
    {dtype = torch.float64}), 1e-4, 1e-4))
  assert_true(y:grad():allclose(torch.tensor({0.8808, 0.8808},
    {dtype = torch.float64}), 1e-4, 1e-4))

  assert_error(function()
    torch.tensor({1, 2}, {dtype = torch.int64}):logaddexp(
      torch.tensor({3, 4}, {dtype = torch.int64}))
  end, "float")
  assert_error(function()
    x:logaddexp(torch.tensor({3, 4}, {dtype = torch.float32}))
  end, "dtype")
  assert_error(function() torch.logaddexp(x, 1) end, "expects a tensor")
  assert_error(function() torch.logaddexp2(x, 1) end, "expects a tensor")
end)

test("hypot handles zeros, infinities, batches, and broadcasting", function()
  local x = torch.tensor({3, 0, math.huge, -math.huge},
    {dtype = torch.float64})
  local y = torch.tensor({4, 0, math.huge, 1}, {dtype = torch.float64})
  assert_eq(x:hypot(y):totable(), {5, 0, math.huge, math.huge})
  assert_eq(torch.hypot(torch.tensor({3, 1}), torch.tensor({4, 0}))
    :totable(), {5, 1})
  assert_shape(torch.tensor({{3, 4}}):hypot(torch.tensor({4, 3})),
    {1, 2})
  local nan = torch.tensor({0 / 0}, {dtype = torch.float64})
  assert_true(nan:hypot(torch.tensor({2}, {dtype = torch.float64}))
    :get(1) ~= nan:hypot(torch.tensor({2}, {dtype = torch.float64}))
    :get(1))
  -- C99 hypot(inf, nan) is inf.
  assert_eq(torch.tensor({math.huge}, {dtype = torch.float64})
    :hypot(nan):get(1), math.huge)
end)

test("hypot is differentiable and validates operands", function()
  local x = torch.tensor({1, 2},
    {dtype = torch.float64, requires_grad = true})
  local y = torch.tensor({3, 4},
    {dtype = torch.float64, requires_grad = true})
  x:hypot(y):sum():backward()
  assert_true(x:grad():allclose(torch.tensor({1 / math.sqrt(10),
    2 / math.sqrt(20)}, {dtype = torch.float64}), 1e-12, 1e-12))
  assert_true(y:grad():allclose(torch.tensor({3 / math.sqrt(10),
    4 / math.sqrt(20)}, {dtype = torch.float64}), 1e-12, 1e-12))

  assert_error(function()
    torch.tensor({1, 2}, {dtype = torch.int64}):hypot(
      torch.tensor({3, 4}, {dtype = torch.int64}))
  end, "float")
  assert_error(function()
    x:hypot(torch.tensor({3, 4}, {dtype = torch.float32}))
  end, "dtype")
  assert_error(function() torch.hypot(x, 1) end, "expects a tensor")
end)

test("frac returns the fractional part for floats and zero for integers", function()
  local values = torch.tensor({1.25, -1.25, 2.5, 0},
    {dtype = torch.float64})
  assert_eq(values:frac():totable(), {0.25, -0.25, 0.5, 0})
  assert_eq(torch.frac(values):totable(), values:frac():totable())
  assert_eq(torch.tensor({1, 2}, {dtype = torch.int64}):frac()
    :totable(), {0, 0})
  assert_error(function() torch.tensor({true, false}):frac() end, "bool")
  assert_error(function() torch.frac(1) end, "expects a tensor")
end)

test("frac is differentiable", function()
  local values = torch.tensor({1.25, 2.5},
    {dtype = torch.float64, requires_grad = true})
  values:frac():sum():backward()
  assert_true(values:grad():allclose(torch.ones(2,
    {dtype = torch.float64})))
end)
