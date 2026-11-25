-- Activation API gaps: softplus, elu, selu, silu, and gelu.

local torch = require("torch")

test("softplus matches log1p(exp(x)) and switches to the linear branch", function()
  local values = torch.tensor({-2, -1, 0, 1, 2}, {dtype = torch.float64})
  local result = values:softplus()
  assert_near(result:get(1), math.log(1 + math.exp(-2)), 1e-12)
  assert_near(result:get(3), math.log(2), 1e-12)
  assert_near(result:get(5), math.log(1 + math.exp(2)), 1e-12)
  assert_eq(torch.softplus(values):get(3), result:get(3))

  local beta_two = values:softplus(2)
  assert_near(beta_two:get(4), math.log(1 + math.exp(2)) / 2, 1e-12)
  assert_eq(torch.tensor({1000}, {dtype = torch.float64}):softplus()
    :get(1), 1000)
  assert_eq(torch.tensor({-1000}, {dtype = torch.float64}):softplus()
    :get(1), 0)
end)

test("softplus is differentiable and validates its options", function()
  local values = torch.tensor({-1, 0.5},
    {dtype = torch.float64, requires_grad = true})
  values:softplus():sum():backward()
  assert_true(values:grad():allclose(
    torch.tensor({1 / (1 + math.exp(1)), 1 / (1 + math.exp(-0.5))},
      {dtype = torch.float64}), 1e-12, 1e-12))
  assert_error(function()
    torch.tensor({1}, {dtype = torch.int64}):softplus()
  end, "float")
  assert_error(function() torch.tensor({1}):softplus(0) end, "beta")
  assert_error(function() torch.tensor({1}):softplus(nil, math.huge)
  end, "threshold")
  assert_error(function() torch.softplus(1) end, "expects a tensor")
end)

test("elu uses alpha times expm1 on the non-positive side", function()
  local values = torch.tensor({-2, -1, 0, 1, 2}, {dtype = torch.float64})
  local result = values:elu()
  assert_near(result:get(1), math.exp(-2) - 1, 1e-12)
  assert_eq(result:get(3), 0)
  assert_eq(result:get(4), 1)
  assert_near(values:elu(2):get(2), 2 * (math.exp(-1) - 1), 1e-12)
  assert_eq(torch.elu(values):get(2), result:get(2))
end)

test("elu is differentiable and selu applies its fixed scale", function()
  local values = torch.tensor({-1, 1},
    {dtype = torch.float64, requires_grad = true})
  values:elu():sum():backward()
  assert_true(values:grad():allclose(
    torch.tensor({math.exp(-1), 1}, {dtype = torch.float64}), 1e-12, 1e-12))

  local selu_input = torch.tensor({-1, 0, 1}, {dtype = torch.float64})
  local selu = selu_input:selu()
  local alpha = 1.6732632423543772
  local scale = 1.0507009873554805
  assert_near(selu:get(1), scale * alpha * (math.exp(-1) - 1), 1e-12)
  assert_eq(selu:get(2), 0)
  assert_near(selu:get(3), scale, 1e-12)
  assert_eq(torch.selu(selu_input):get(3), selu:get(3))
end)

test("selu is differentiable", function()
  local values = torch.tensor({-1, 1},
    {dtype = torch.float64, requires_grad = true})
  values:selu():sum():backward()
  local alpha = 1.6732632423543772
  local scale = 1.0507009873554805
  assert_true(values:grad():allclose(
    torch.tensor({scale * alpha * math.exp(-1), scale},
      {dtype = torch.float64}), 1e-12, 1e-12))
end)

test("silu multiplies the input by its sigmoid", function()
  local values = torch.tensor({-1, 0, 1, 2}, {dtype = torch.float64})
  local result = values:silu()
  assert_near(result:get(1), -1 / (1 + math.exp(1)), 1e-12)
  assert_eq(result:get(2), 0)
  assert_near(result:get(3), 1 / (1 + math.exp(-1)), 1e-12)
  assert_eq(torch.silu(values):get(3), result:get(3))

  local grad_values = torch.tensor({-1, 1},
    {dtype = torch.float64, requires_grad = true})
  grad_values:silu():sum():backward()
  local sigmoid = 1 / (1 + math.exp(-1))
  local other = 1 / (1 + math.exp(1))
  assert_true(grad_values:grad():allclose(torch.tensor({
    other + (-1) * other * (1 - other), sigmoid + sigmoid * (1 - sigmoid)},
    {dtype = torch.float64}), 1e-12, 1e-12))
end)

test("gelu supports the exact and tanh forms", function()
  local values = torch.tensor({-1, 0, 1, 2}, {dtype = torch.float64})
  local exact = values:gelu()
  assert_near(exact:get(1), -0.15865525393145707, 1e-12)
  assert_eq(exact:get(2), 0)
  assert_near(exact:get(3), 0.8413447460685429, 1e-12)
  assert_near(exact:get(4), 1.9544997361036416, 1e-12)

  local tanh_form = values:gelu("tanh")
  assert_near(tanh_form:get(1), -0.1588080093180125, 1e-6)
  assert_near(tanh_form:get(3), 0.8411919906082768, 1e-6)
  assert_eq(torch.gelu(values):get(3), exact:get(3))
  assert_eq(torch.gelu(values, "tanh"):get(3), tanh_form:get(3))
end)

test("gelu is differentiable in both forms and validates options", function()
  local values = torch.tensor({0, 1},
    {dtype = torch.float64, requires_grad = true})
  values:gelu():sum():backward()
  assert_true(values:grad():allclose(
    torch.tensor({0.5, 1.0833154705876864},
      {dtype = torch.float64}), 1e-12, 1e-12))

  local tanh_values = torch.tensor({0, 1},
    {dtype = torch.float64, requires_grad = true})
  tanh_values:gelu("tanh"):sum():backward()
  assert_true(tanh_values:grad():allclose(
    torch.tensor({0.5, 1.0829640838457828}, {dtype = torch.float64}),
    1e-5, 1e-5))

  assert_error(function() torch.tensor({1}):gelu("fast") end, "approximate")
  assert_error(function() torch.gelu(1) end, "expects a tensor")
end)

test("activations preserve float16 and reject bool tensors", function()
  local half = torch.tensor({-1, 0, 1}, {dtype = torch.float16})
  assert_eq(half:softplus():dtype(), "float16")
  assert_eq(half:elu():dtype(), "float16")
  assert_eq(half:selu():dtype(), "float16")
  assert_eq(half:silu():dtype(), "float16")
  assert_eq(half:gelu():dtype(), "float16")
  assert_error(function()
    torch.tensor({true, false}):softplus()
  end, "float")
  assert_error(function() torch.tensor({true, false}):gelu() end, "float")
end)
