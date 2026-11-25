-- Unary math gaps: log2, log10, expm1, erf/erfc, and normalized sinc.

local torch = require("torch")

test("log2 and log10 compute base-2 and base-10 logarithms", function()
  local powers = torch.tensor({0.25, 1, 4, 8}, {dtype = torch.float64})
  assert_eq(powers:log2():totable(), {-2, 0, 2, 3})
  local decades = torch.tensor({0.01, 1, 10, 100}, {dtype = torch.float64})
  assert_eq(decades:log10():totable(), {-2, 0, 1, 2})
  assert_eq(torch.log2(powers):totable(), powers:log2():totable())
  assert_eq(torch.log10(decades):totable(), decades:log10():totable())
end)

test("log2 and log10 support in-place calls and float16", function()
  local values = torch.tensor({4, 8}, {dtype = torch.float64})
  assert_eq(values:log2_():totable(), {2, 3})
  local half = torch.tensor({4, 8}, {dtype = torch.float16})
  assert_eq(half:log2():totable(), {2, 3})
  assert_eq(half:log2():dtype(), "float16")
end)

test("expm1 keeps precision near zero", function()
  local values = torch.tensor({1e-8, 0, 1}, {dtype = torch.float64})
  local result = values:expm1()
  assert_near(result:get(1), 1e-8, 1e-15)
  assert_eq(result:get(2), 0)
  assert_near(result:get(3), math.exp(1) - 1, 1e-12)
  assert_eq(torch.expm1(values):get(1), result:get(1))
  assert_near(torch.expm1(values):get(3), math.exp(1) - 1, 1e-12)
end)

test("log2, log10, and expm1 are differentiable", function()
  local log2_input = torch.tensor({0.5, 1.5},
    {dtype = torch.float64, requires_grad = true})
  log2_input:log2():sum():backward()
  assert_true(log2_input:grad():allclose(
    torch.tensor({1 / (0.5 * math.log(2)), 1 / (1.5 * math.log(2))},
      {dtype = torch.float64}), 1e-12, 1e-12))

  local log10_input = torch.tensor({0.5, 1.5},
    {dtype = torch.float64, requires_grad = true})
  log10_input:log10():sum():backward()
  assert_true(log10_input:grad():allclose(
    torch.tensor({1 / (0.5 * math.log(10)), 1 / (1.5 * math.log(10))},
      {dtype = torch.float64}), 1e-12, 1e-12))

  local exp_input = torch.tensor({0.5, 1.5},
    {dtype = torch.float64, requires_grad = true})
  exp_input:expm1():sum():backward()
  assert_true(exp_input:grad():allclose(
    torch.tensor({math.exp(0.5), math.exp(1.5)},
      {dtype = torch.float64}), 1e-12, 1e-12))
end)

test("erf and erfc match their known values and complement identity", function()
  local values = torch.tensor({0, 0.5, 1}, {dtype = torch.float64})
  local erf = values:erf()
  local erfc = values:erfc()
  assert_eq(erf:get(1), 0)
  assert_near(erf:get(2), 0.5204998778130465, 1e-12)
  assert_near(erf:get(3), 0.8427007929497149, 1e-12)
  assert_near(erfc:get(2), 0.4795001221869535, 1e-12)
  assert_true(erf:add(erfc):allclose(torch.ones(3,
    {dtype = torch.float64}), 1e-12, 1e-12))
  assert_true(torch.tensor({-1, 1}, {dtype = torch.float64}):erf()
    :add(torch.tensor({1, -1}, {dtype = torch.float64}):erf())
    :allclose(torch.zeros(2, {dtype = torch.float64}), 1e-12, 1e-12))
  assert_eq(torch.erf(values):get(2), erf:get(2))
  assert_eq(torch.erfc(values):get(2), erfc:get(2))
end)

test("erf and erfc are differentiable", function()
  local values = torch.tensor({0.5, 1},
    {dtype = torch.float64, requires_grad = true})
  values:erf():sum():backward()
  local scale = 2 / math.sqrt(math.pi)
  assert_true(values:grad():allclose(torch.tensor({
    scale * math.exp(-0.25), scale * math.exp(-1)},
    {dtype = torch.float64}), 1e-12, 1e-12))

  local complement = torch.tensor({0.5, 1},
    {dtype = torch.float64, requires_grad = true})
  complement:erfc():sum():backward()
  assert_true(complement:grad():allclose(torch.tensor({
    -scale * math.exp(-0.25), -scale * math.exp(-1)},
    {dtype = torch.float64}), 1e-12, 1e-12))
end)

test("sinc uses the normalized sinc convention", function()
  local values = torch.tensor({0, 0.5, 1, -1}, {dtype = torch.float64})
  local result = values:sinc()
  assert_eq(result:get(1), 1)
  assert_near(result:get(2), 2 / math.pi, 1e-12)
  assert_near(result:get(3), 0, 1e-15)
  assert_near(result:get(4), 0, 1e-15)
  assert_eq(torch.sinc(values):get(1), 1)
end)

test("sinc is differentiable and supports in-place calls", function()
  local values = torch.tensor({0, 1}, {dtype = torch.float64})
  assert_eq(values:sinc_():get(1), 1)
  local grad_input = torch.tensor({1},
    {dtype = torch.float64, requires_grad = true})
  grad_input:sinc():sum():backward()
  assert_near(grad_input:grad():get(1), -1, 1e-12)
end)
