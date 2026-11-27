-- Binary elementwise gaps: fmod, floor_divide, atan2, copysign, heaviside.

local torch = require("torch")

test("fmod truncates toward zero and differs from remainder", function()
  local dividend = torch.tensor({5, -5, 5, -5}, {dtype = torch.float64})
  local divisor = torch.tensor({3, 3, -3, -3}, {dtype = torch.float64})
  assert_eq(dividend:fmod(divisor):totable(), {2, -2, 2, -2})
  assert_eq(dividend:remainder(divisor):totable(), {2, 1, -1, -2})
  assert_eq(torch.fmod(dividend, divisor):totable(), {2, -2, 2, -2})
  local values = torch.tensor({5}, {dtype = torch.float64})
  assert_eq(values:fmod_(torch.tensor({3}, {dtype = torch.float64}))
    :totable(), {2})
  assert_error(function() torch.fmod(1, divisor) end, "first argument")
end)

test("floor_divide rounds toward negative infinity", function()
  local dividend = torch.tensor({7, -7, 7, -7}, {dtype = torch.float64})
  local divisor = torch.tensor({3, 3, -3, -3}, {dtype = torch.float64})
  assert_eq(dividend:floor_divide(divisor):totable(), {2, -3, -3, 2})
  assert_eq(torch.floor_divide(dividend, divisor):totable(),
    {2, -3, -3, 2})

  local integers = torch.tensor({7, -7}, {dtype = torch.int64})
  assert_eq(integers:floor_divide(torch.tensor({3, 3},
    {dtype = torch.int64})):totable(), {2, -3})
  assert_eq(integers:floor_divide(torch.tensor({3, 3},
    {dtype = torch.int64})):dtype(), "int64")
  local values = torch.tensor({7}, {dtype = torch.float64})
  assert_eq(values:floor_divide_(torch.tensor({2},
    {dtype = torch.float64})):totable(), {3})
end)

test("atan2 covers every quadrant and broadcasts", function()
  local y = torch.tensor({0, 1, 0, -1, 1}, {dtype = torch.float64})
  local x = torch.tensor({1, 0, -1, 0, 1}, {dtype = torch.float64})
  local result = y:atan2(x)
  assert_near(result:get(1), 0, 1e-12)
  assert_near(result:get(2), math.pi / 2, 1e-12)
  assert_near(result:get(3), math.pi, 1e-12)
  assert_near(result:get(4), -math.pi / 2, 1e-12)
  assert_near(result:get(5), math.pi / 4, 1e-12)
  assert_eq(torch.atan2(y, x):get(2), result:get(2))

  local rows = torch.tensor({{0, 1}}, {dtype = torch.float64})
  assert_shape(rows:atan2(torch.tensor({{1}, {0}},
    {dtype = torch.float64})), {2, 2})
end)

test("copysign copies the magnitude and sign separately", function()
  local magnitudes = torch.tensor({3, -3, 0, 0}, {dtype = torch.float64})
  local signs = torch.tensor({-2, 2, -2, 2}, {dtype = torch.float64})
  local result = magnitudes:copysign(signs)
  assert_eq(result:totable(), {-3, 3, 0, 0})
  assert_eq(torch.copysign(magnitudes, signs):totable(), {-3, 3, 0, 0})
  -- -0 preserves the negative sign.
  assert_eq(torch.tensor({1}, {dtype = torch.float64}):div(result):get(3),
    -math.huge)
end)

test("heaviside uses the supplied value at zero", function()
  local values = torch.tensor({-1, 0, 1}, {dtype = torch.float64})
  assert_eq(values:heaviside(0.5):totable(), {0, 0.5, 1})
  assert_eq(values:heaviside(torch.tensor({0.25, 0.5, 0.75},
    {dtype = torch.float64})):totable(), {0, 0.5, 1})
  assert_eq(torch.heaviside(values, 2):totable(), {0, 2, 1})

  local in_place = torch.tensor({-1, 0, 1}, {dtype = torch.float64})
  local result = in_place:heaviside_(0.5)
  assert_true(result == in_place)
  assert_eq(in_place:totable(), {0, 0.5, 1})
end)

test("heaviside validates its replacement values", function()
  local input = torch.tensor({0}, {dtype = torch.float64})
  assert_error(function() input:heaviside("x") end, "number or a tensor")
  assert_error(function() input:heaviside(torch.tensor({0.5},
    {dtype = torch.float32})) end, "dtype")
  assert_error(function() torch.heaviside(1, 0.5) end, "first argument")
end)

test("new binary ops preserve float16 and reject non-tensors", function()
  local half = torch.tensor({5, -5}, {dtype = torch.float16})
  local remainder = half:fmod(torch.tensor({3, 3},
    {dtype = torch.float16}))
  assert_eq(remainder:dtype(), "float16")
  assert_eq(remainder:totable(), {2, -2})
  assert_error(function() half:atan2(1) end, "Tensor or a number")
  assert_error(function() half:copysign("x") end, "Tensor or a number")
  assert_error(function() half:floor_divide(nil) end, "Tensor or a number")
end)
