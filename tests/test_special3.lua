-- Additional special functions: log_ndtr, ndtri, i0e, i1, i1e.

local torch = require("torch")

test("log_ndtr is the log of the normal CDF", function()
  local values = torch.tensor({-1, 0, 1}, {dtype = torch.float64})
  local result = values:log_ndtr()
  assert_near(result:get(1), math.log(0.15865525393145707), 1e-12)
  assert_near(result:get(2), math.log(0.5), 1e-12)
  assert_near(result:get(3), math.log(0.8413447460685429), 1e-12)
  assert_near(result:get(3), values:ndtr():log():get(3), 1e-12)
  assert_near(torch.log_ndtr(values):get(2), math.log(0.5), 1e-12)
end)

test("ndtri is the inverse normal CDF", function()
  local probabilities = torch.tensor({0.025, 0.5, 0.975},
    {dtype = torch.float64})
  local result = probabilities:ndtri()
  assert_near(result:get(1), -1.959963984540054, 1e-12)
  assert_eq(result:get(2), 0)
  assert_near(result:get(3), 1.959963984540054, 1e-12)
  assert_eq(torch.ndtri(probabilities):get(2), 0)

  local inputs = torch.tensor({-1, 0.5, 1}, {dtype = torch.float64})
  assert_true(inputs:ndtr():ndtri():allclose(inputs, 1e-12, 1e-12))
end)

test("i0e scales i0 by exp(-abs(x))", function()
  local values = torch.tensor({0, 1, 2}, {dtype = torch.float64})
  local scaled = values:i0e()
  assert_eq(scaled:get(1), 1)
  assert_near(scaled:get(2), 0.46575960759364043, 1e-12)
  assert_near(scaled:get(2), values:i0():get(2) * math.exp(-1), 1e-12)
  assert_near(scaled:get(3), values:i0():get(3) * math.exp(-2), 1e-12)
  assert_eq(torch.i0e(values):get(1), 1)
end)

test("i1 evaluates the order-one modified Bessel function", function()
  local values = torch.tensor({0, 1, 2}, {dtype = torch.float64})
  local result = values:i1()
  assert_eq(result:get(1), 0)
  assert_near(result:get(2), 0.565159103992485, 1e-12)
  assert_eq(torch.i1(values):get(1), 0)
  assert_near(torch.i1(values):get(2), result:get(2), 1e-12)
end)

test("i1e scales i1 by exp(-abs(x))", function()
  local values = torch.tensor({0, 1, 2}, {dtype = torch.float64})
  local scaled = values:i1e()
  assert_eq(scaled:get(1), 0)
  assert_near(scaled:get(2), 0.2079104153497085, 1e-12)
  assert_near(scaled:get(2), values:i1():get(2) * math.exp(-1), 1e-12)
  assert_near(scaled:get(3), values:i1():get(3) * math.exp(-2), 1e-12)
  assert_eq(torch.i1e(values):get(1), 0)
end)

test("special functions are differentiable", function()
  local ndtri_input = torch.tensor({0.5}, {dtype = torch.float64,
    requires_grad = true})
  ndtri_input:ndtri():sum():backward()
  assert_near(ndtri_input:grad():get(1),
    math.sqrt(2 * math.pi), 1e-12)

  local log_input = torch.tensor({0.5}, {dtype = torch.float64,
    requires_grad = true})
  log_input:log_ndtr():sum():backward()
  assert_near(log_input:grad():get(1),
    0.3520653267642995 / 0.6914624612740131, 1e-12)
end)

test("in-place variants and aliases work", function()
  local values = torch.tensor({0.5}, {dtype = torch.float64})
  assert_eq(values:i0e_():dtype(), "float64")
  local ndtri_values = torch.tensor({0.5}, {dtype = torch.float64})
  assert_eq(ndtri_values:ndtri_():get(1), 0)
  assert_error(function() torch.log_ndtr(1) end, "index")
  assert_error(function() torch.ndtri(1) end, "index")
  assert_error(function() torch.i0e(1) end, "index")
  assert_error(function() torch.i1(1) end, "index")
  assert_error(function() torch.i1e(1) end, "index")
end)
