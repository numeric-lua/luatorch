-- Special functions: erfinv, i0, lgamma, digamma, ndtr.

local torch = require("torch")

test("erfinv inverts erf", function()
  local values = torch.tensor({0, 0.8427007929497149},
    {dtype = torch.float64})
  local inverse = values:erfinv()
  assert_eq(inverse:get(1), 0)
  assert_near(inverse:get(2), 1, 1e-12)
  assert_near(torch.erf(values):erfinv():get(2), values:get(2), 1e-12)
  assert_eq(torch.erfinv(values):get(1), 0)

  local in_place = torch.tensor({0.5}, {dtype = torch.float64})
  assert_near(in_place:erfinv_():get(1), 0.4769362762044699, 1e-12)
  assert_near(torch.tensor({1}, {dtype = torch.float64}):erfinv():get(1),
    math.huge)
end)

test("i0 evaluates the modified Bessel function", function()
  local values = torch.tensor({0, 1, 2}, {dtype = torch.float64})
  local result = values:i0()
  assert_eq(result:get(1), 1)
  assert_near(result:get(2), 1.2660658777520082, 1e-12)
  assert_near(result:get(3), 2.2795853023360673, 1e-12)
  assert_near(torch.i0(values):get(2), result:get(2), 1e-12)
  assert_eq(torch.tensor({0}, {dtype = torch.float16}):i0():dtype(),
    "float16")
end)

test("lgamma matches log-gamma identities", function()
  local values = torch.tensor({1, 2, 5}, {dtype = torch.float64})
  local result = values:lgamma()
  assert_eq(result:get(1), 0)
  assert_eq(result:get(2), 0)
  assert_near(result:get(3), math.log(24), 1e-12)
  assert_near(torch.lgamma(values):get(3), math.log(24), 1e-12)
end)

test("digamma satisfies the recurrence", function()
  local values = torch.tensor({1, 2, 3}, {dtype = torch.float64})
  local result = values:digamma()
  local euler = 0.5772156649015329
  assert_near(result:get(1), -euler, 1e-12)
  assert_near(result:get(2), 1 - euler, 1e-12)
  assert_near(result:get(2) - result:get(1), 1, 1e-12)
  assert_near(torch.digamma(values):get(1), -euler, 1e-12)
end)

test("ndtr is the standard normal CDF", function()
  local values = torch.tensor({-1, 0, 1}, {dtype = torch.float64})
  local result = values:ndtr()
  assert_near(result:get(1), 0.15865525393145707, 1e-12)
  assert_eq(result:get(2), 0.5)
  assert_near(result:get(3), 0.8413447460685429, 1e-12)
  assert_near(result:get(1) + result:get(3), 1, 1e-12)
  assert_eq(torch.ndtr(values):get(2), 0.5)
end)

test("special functions are differentiable", function()
  local values = torch.tensor({0.5}, {dtype = torch.float64,
    requires_grad = true})
  values:ndtr():sum():backward()
  assert_near(values:grad():get(1), 0.3520653267642995, 1e-12)

  local erfinv_input = torch.tensor({0.5}, {dtype = torch.float64,
    requires_grad = true})
  erfinv_input:erfinv():sum():backward()
  assert_true(erfinv_input:grad():get(1) == erfinv_input:grad():get(1))
end)

test("special function aliases validate tensors", function()
  assert_error(function() torch.erfinv(1) end, "index")
  assert_error(function() torch.i0(1) end, "index")
  assert_error(function() torch.lgamma(1) end, "index")
  assert_error(function() torch.digamma(1) end, "index")
  assert_error(function() torch.ndtr(1) end, "index")
end)
