-- Additional activation and angle helpers: deg2rad, rad2deg, softsign,
-- hardsigmoid, hardswish.

local torch = require("torch")

test("deg2rad and rad2deg convert angles and round-trip", function()
  local degrees = torch.tensor({0, 90, 180, 360}, {dtype = torch.float64})
  local radians = degrees:deg2rad()
  assert_near(radians:get(2), math.pi / 2, 1e-12)
  assert_near(radians:get(3), math.pi, 1e-12)
  assert_near(radians:get(4), 2 * math.pi, 1e-12)
  assert_near(radians:rad2deg():get(3), 180, 1e-12)
  assert_near(torch.deg2rad(degrees):get(2), math.pi / 2, 1e-12)
  assert_near(torch.rad2deg(torch.tensor({math.pi}, {dtype = torch.float64}))
    :get(1), 180, 1e-12)
end)

test("angle conversions are differentiable", function()
  local degrees = torch.tensor({180},
    {dtype = torch.float64, requires_grad = true})
  degrees:deg2rad():backward()
  assert_near(degrees:grad():get(1), math.pi / 180, 1e-12)

  local radians = torch.tensor({math.pi},
    {dtype = torch.float64, requires_grad = true})
  radians:rad2deg():backward()
  assert_near(radians:grad():get(1), 180 / math.pi, 1e-12)
end)

test("softsign maps values to the open interval", function()
  local values = torch.tensor({-3, 0, 3}, {dtype = torch.float64})
  assert_eq(values:softsign():totable(), {-0.75, 0, 0.75})
  assert_eq(torch.softsign(values):totable(), {-0.75, 0, 0.75})

  local grad_values = torch.tensor({-3, 0, 3},
    {dtype = torch.float64, requires_grad = true})
  grad_values:softsign():sum():backward()
  assert_eq(grad_values:grad():totable(), {0.0625, 1, 0.0625})
end)

test("hardsigmoid clamps to [0, 1] with a linear middle", function()
  local values = torch.tensor({-4, -3, -1, 0, 1, 3, 4},
    {dtype = torch.float64})
  local result = values:hardsigmoid()
  assert_eq(result:get(1), 0)
  assert_eq(result:get(2), 0)
  assert_near(result:get(3), 1 / 3, 1e-12)
  assert_eq(result:get(4), 0.5)
  assert_near(result:get(5), 2 / 3, 1e-12)
  assert_eq(result:get(6), 1)
  assert_eq(result:get(7), 1)
  assert_eq(torch.hardsigmoid(values):get(4), 0.5)
end)

test("hardsigmoid gradient is 1/6 inside and zero outside", function()
  local values = torch.tensor({-4, -1, 0, 1, 4},
    {dtype = torch.float64, requires_grad = true})
  values:hardsigmoid():sum():backward()
  assert_true(values:grad():allclose(
    torch.tensor({0, 1 / 6, 1 / 6, 1 / 6, 0},
      {dtype = torch.float64}), 1e-12, 1e-12))
end)

test("hardswish multiplies the input by hardsigmoid", function()
  local values = torch.tensor({-4, -3, -1, 0, 1, 3, 4},
    {dtype = torch.float64})
  local result = values:hardswish()
  assert_eq(result:get(1), 0)
  assert_eq(result:get(2), 0)
  assert_near(result:get(3), -1 / 3, 1e-12)
  assert_eq(result:get(4), 0)
  assert_near(result:get(5), 2 / 3, 1e-12)
  assert_eq(result:get(6), 3)
  assert_eq(result:get(7), 4)
  assert_eq(torch.hardswish(values):get(5), result:get(5))

  local grad_values = torch.tensor({-1, 0, 1},
    {dtype = torch.float64, requires_grad = true})
  grad_values:hardswish():sum():backward()
  assert_true(grad_values:grad():allclose(
    torch.tensor({1 / 6, 0.5, 5 / 6}, {dtype = torch.float64}),
    1e-12, 1e-12))
end)

test("new activation helpers preserve float16 and reject non-floating inputs", function()
  local half = torch.tensor({-1, 0, 1}, {dtype = torch.float16})
  assert_eq(half:deg2rad():dtype(), "float16")
  assert_eq(half:rad2deg():dtype(), "float16")
  assert_eq(half:softsign():dtype(), "float16")
  assert_eq(half:hardsigmoid():dtype(), "float16")
  assert_eq(half:hardswish():dtype(), "float16")

  assert_error(function()
    torch.tensor({1, 2}, {dtype = torch.int64}):softsign()
  end, "float")
  assert_error(function() torch.tensor({true}):hardsigmoid() end, "float")
  assert_error(function() torch.deg2rad(1) end, "expects a tensor")
  assert_error(function() torch.hardswish(1) end, "expects a tensor")
end)
