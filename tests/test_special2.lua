-- Cumulative extrema and special elementwise helpers: cummax, cummin, entr,
-- xlogy, xlog1py.

local torch = require("torch")

test("cummax returns running maxima and 1-based indices", function()
  local values = torch.tensor({1, 3, 2, 5, 4}, {dtype = torch.float64})
  local maxima, indices = values:cummax()
  assert_eq(maxima:totable(), {1, 3, 3, 5, 5})
  assert_eq(indices:totable(), {1, 2, 2, 4, 4})
  assert_eq(indices:dtype(), "int64")

  local matrix = torch.tensor({{1, 4}, {3, 2}}, {dtype = torch.float64})
  local max1, arg1 = matrix:cummax(1)
  assert_eq(max1:totable(), {{1, 4}, {3, 4}})
  assert_eq(arg1:totable(), {{1, 1}, {2, 1}})
  local max2 = matrix:cummax(2)
  assert_eq(max2:totable(), {{1, 4}, {3, 3}})
  local alias_values = torch.cummax(values)
  assert_eq(alias_values:totable(), maxima:totable())
end)

test("cummin returns running minima and 1-based indices", function()
  local values = torch.tensor({5, 3, 4, 1, 2}, {dtype = torch.float64})
  local minima, indices = values:cummin()
  assert_eq(minima:totable(), {5, 3, 3, 1, 1})
  assert_eq(indices:totable(), {1, 2, 2, 4, 4})
  local matrix = torch.tensor({{1, 4}, {3, 2}}, {dtype = torch.float64})
  local min1 = matrix:cummin(1)
  assert_eq(min1:totable(), {{1, 4}, {1, 2}})
  local min2 = matrix:cummin(2)
  assert_eq(min2:totable(), {{1, 1}, {3, 2}})
  local alias_values = torch.cummin(values)
  assert_eq(alias_values:totable(), minima:totable())
end)

test("cummax is differentiable and validates dimensions", function()
  local values = torch.tensor({1, 3, 2},
    {dtype = torch.float64, requires_grad = true})
  local maxima = values:cummax()
  maxima:sum():backward()
  assert_eq(values:grad():totable(), {1, 2, 0})

  assert_error(function() torch.cummax(1) end, "expects a tensor")
  assert_error(function() torch.cummin(1) end, "expects a tensor")
  assert_error(function() values:cummax(3) end, "dimension")
end)

test("entr matches -x log(x) with boundary values", function()
  local nan = 0 / 0
  local values = torch.tensor({-1, 0, 0.5, 1, nan},
    {dtype = torch.float64})
  local result = values:entr()
  assert_eq(result:get(1), -math.huge)
  assert_eq(result:get(2), 0)
  assert_near(result:get(3), -0.5 * math.log(0.5), 1e-12)
  assert_eq(result:get(4), 0)
  assert_true(result:get(5) ~= result:get(5))
  assert_eq(torch.entr(values):get(2), 0)
end)

test("entr is differentiable and floating-only", function()
  local values = torch.tensor({0.5, 1},
    {dtype = torch.float64, requires_grad = true})
  values:entr():sum():backward()
  assert_true(values:grad():allclose(torch.tensor({
    -math.log(0.5) - 1, -1}, {dtype = torch.float64}), 1e-12, 1e-12))

  assert_error(function()
    torch.tensor({1, 2}, {dtype = torch.int64}):entr()
  end, "float")
  assert_error(function() torch.entr(1) end, "expects a tensor")
end)

test("xlogy computes x log(y) with the zero convention", function()
  local x = torch.tensor({0, 1, 2}, {dtype = torch.float64})
  local y = torch.tensor({0, 1, 2}, {dtype = torch.float64})
  local result = x:xlogy(y)
  assert_eq(result:get(1), 0)
  assert_eq(result:get(2), 0)
  assert_near(result:get(3), 2 * math.log(2), 1e-12)
  assert_eq(torch.xlogy(x, y):get(3), result:get(3))
end)

test("xlogy is differentiable away from the singular zero point", function()
  local x = torch.tensor({1}, {dtype = torch.float64, requires_grad = true})
  local y = torch.tensor({2}, {dtype = torch.float64, requires_grad = true})
  x:xlogy(y):sum():backward()
  assert_near(x:grad():get(1), math.log(2), 1e-12)
  assert_near(y:grad():get(1), 0.5, 1e-12)

  assert_error(function() x:xlogy(1) end, "expects a tensor")
  assert_error(function() x:xlogy(torch.tensor({2},
    {dtype = torch.float32})) end, "dtype")
end)

test("xlog1py computes x log1p(y) with the zero convention", function()
  local x = torch.tensor({0, 1, 2}, {dtype = torch.float64})
  local y = torch.tensor({0, 1, 2}, {dtype = torch.float64})
  local result = x:xlog1py(y)
  assert_eq(result:get(1), 0)
  assert_near(result:get(2), math.log(2), 1e-12)
  assert_near(result:get(3), 2 * math.log(3), 1e-12)
  assert_eq(torch.xlog1py(x, y):get(3), result:get(3))

  local grad_x = torch.tensor({1}, {dtype = torch.float64,
    requires_grad = true})
  local grad_y = torch.tensor({2}, {dtype = torch.float64,
    requires_grad = true})
  grad_x:xlog1py(grad_y):sum():backward()
  assert_near(grad_x:grad():get(1), math.log(3), 1e-12)
  assert_near(grad_y:grad():get(1), 1 / 3, 1e-12)
end)
