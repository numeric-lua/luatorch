-- Activation modules: Softplus, ELU, SELU, SiLU, GELU.

local torch = require("torch")
local nn = require("torch.nn")

test("Softplus applies the default and custom beta/threshold", function()
  local values = torch.tensor({-1, 0, 1}, {dtype = torch.float64})
  local output = nn.Softplus()(values)
  assert_near(output:get(1), 0.31326168751822286, 1e-12)
  assert_near(output:get(2), math.log(2), 1e-12)
  assert_near(output:get(3), 1.3132616875182228, 1e-12)

  local scaled = nn.Softplus(2)(values)
  assert_near(scaled:get(2), math.log(2) / 2, 1e-12)
  assert_eq(tostring(nn.Softplus(2, 10)),
    "Softplus(beta=2, threshold=10)")
end)

test("ELU uses alpha and supports in-place evaluation", function()
  local values = torch.tensor({-1, 0, 1}, {dtype = torch.float64})
  local output = nn.ELU()(values)
  assert_near(output:get(1), math.exp(-1) - 1, 1e-12)
  assert_eq(output:get(2), 0)
  assert_eq(output:get(3), 1)

  local alpha = nn.ELU(2)(values)
  assert_near(alpha:get(1), 2 * (math.exp(-1) - 1), 1e-12)

  local in_place = torch.tensor({-1}, {dtype = torch.float64})
  local result = nn.ELU(1, true)(in_place)
  assert_true(result == in_place)
  assert_near(in_place:get(1), math.exp(-1) - 1, 1e-12)
end)

test("SELU applies the fixed scale and alpha", function()
  local values = torch.tensor({-1, 0, 1}, {dtype = torch.float64})
  local output = nn.SELU()(values)
  local scale, alpha = 1.0507009873554805, 1.6732632423543772
  assert_near(output:get(1), scale * alpha * (math.exp(-1) - 1), 1e-12)
  assert_eq(output:get(2), 0)
  assert_near(output:get(3), scale, 1e-12)
end)

test("SiLU applies x * sigmoid(x)", function()
  local values = torch.tensor({-1, 0, 1}, {dtype = torch.float64})
  local output = nn.SiLU()(values)
  assert_near(output:get(1), -1 / (1 + math.exp(1)), 1e-12)
  assert_eq(output:get(2), 0)
  assert_near(output:get(3), 1 / (1 + math.exp(-1)), 1e-12)
end)

test("GELU supports the exact and tanh approximations", function()
  local values = torch.tensor({-1, 0, 1}, {dtype = torch.float64})
  local exact = nn.GELU()(values)
  assert_near(exact:get(1), -0.15865525393145707, 1e-12)
  assert_near(exact:get(2), 0, 1e-12)
  assert_near(exact:get(3), 0.8413447460685429, 1e-12)

  local approximate = nn.GELU("tanh")(values)
  assert_near(approximate:get(1), -0.158808009433984, 1e-5)
  assert_near(approximate:get(3), 0.841191990566016, 1e-5)
  assert_eq(tostring(nn.GELU()), 'GELU(approximate="none")')
end)

test("activation modules compose with Sequential and autograd", function()
  local model = nn.Sequential(nn.Linear(2, 2), nn.GELU(), nn.SiLU())
  local input = torch.tensor({{1, 2}})
  local output = model(input)
  assert_shape(output, {1, 2})
  output:sum():backward()
  assert_true(model._modules["1"]._parameters.weight:grad() ~= nil)
end)

test("activation modules expose formats and preserve dtype", function()
  local half = torch.tensor({-1, 0, 1}, {dtype = torch.float16})
  assert_eq(nn.Softplus()(half):dtype(), "float16")
  assert_eq(nn.ELU()(half):dtype(), "float16")
  assert_eq(nn.SELU()(half):dtype(), "float16")
  assert_eq(nn.SiLU()(half):dtype(), "float16")
  assert_eq(nn.GELU()(half):dtype(), "float16")
  assert_eq(tostring(nn.ELU(0.5, true)),
    "ELU(alpha=0.5, inplace=true)")
end)
