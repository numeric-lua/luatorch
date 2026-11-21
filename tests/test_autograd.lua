local torch = require("torch")

test("simple gradient", function()
  local x = torch.tensor({ 1, 2, 3 }, { requires_grad = true })
  assert_true(x:requires_grad())
  local loss = x:mul(x):sum()
  loss:backward()
  assert_near(loss:item(), 14)
  assert_tensor(x:grad(), { 2, 4, 6 })
end)

test("chain rule through scalar ops", function()
  local x = torch.tensor({ 1, 2, 3 }, { requires_grad = true })
  local loss = x:mul(2):add(1):pow(2):sum()
  loss:backward()
  -- d/dx (2x + 1)^2 = 8x + 4
  assert_tensor(x:grad(), { 12, 20, 28 })
end)

test("gradient accumulation", function()
  local x = torch.tensor({ 1, 2 }, { requires_grad = true })
  x:mul(2):sum():backward()
  x:mul(3):sum():backward()
  assert_tensor(x:grad(), { 5, 5 })
  x:zero_grad()
  assert_tensor(x:grad(), { 0, 0 })
end)

test("requires_grad_ on an existing tensor", function()
  local x = torch.tensor({ 1, 2 })
  assert_false(x:requires_grad())
  x:requires_grad_(true)
  assert_true(x:requires_grad())
  x:sum():backward()
  assert_tensor(x:grad(), { 1, 1 })
end)

test("backward with an explicit gradient", function()
  local x = torch.tensor({ 1, 2, 3 }, { requires_grad = true })
  local total = x:sum()
  total:backward(torch.tensor(2.0))
  assert_tensor(x:grad(), { 2, 2, 2 })
end)

test("detach stops the graph", function()
  local x = torch.tensor({ 1, 2, 3 }, { requires_grad = true })
  local detached = x:detach()
  assert_false(detached:requires_grad())
  local loss = detached:mul(2):sum()
  assert_false(loss:requires_grad())
  assert_error(function()
    loss:backward()
  end, "does not require grad")
end)

test("non-leaf gradients flow to leaves", function()
  local x = torch.tensor({ 1, 2 }, { requires_grad = true })
  local hidden = x:mul(3)
  assert_true(hidden:requires_grad())
  hidden:sum():backward()
  assert_tensor(x:grad(), { 3, 3 })
end)

test("requires_grad on integer tensors is rejected", function()
  assert_error(function()
    torch.tensor({ 1, 2 }, { dtype = torch.int64, requires_grad = true })
  end, "floating point")
end)

test("in-place on a leaf requiring grad is rejected", function()
  local x = torch.tensor({ 1, 2 }, { requires_grad = true })
  assert_error(function()
    x:add_(1)
  end, "leaf")
end)

test("gradient dtype matches tensor dtype", function()
  local x = torch.tensor({ 1, 2 }, { dtype = torch.float64, requires_grad = true })
  x:mul(x):sum():backward()
  assert_dtype(x:grad(), "float64")
end)

test("no_grad disables graph construction", function()
  local x = torch.tensor({ 1, 2 }, { requires_grad = true })
  local y
  torch.no_grad(function()
    y = x:mul(2)
  end)
  assert_false(y:requires_grad())
  assert_true(torch.is_grad_enabled())

  local z = torch.no_grad(function()
    return x:sum()
  end)
  assert_false(z:requires_grad())
  assert_true(torch.is_grad_enabled())
end)

test("grad is nil before backward", function()
  local x = torch.tensor({ 1, 2 }, { requires_grad = true })
  assert_eq(x:grad(), nil)
end)

test("matrix_power gradients flow through positive powers", function()
  local a = torch.tensor({ { 1, 2 }, { 3, 4 } },
    { dtype = torch.float64, requires_grad = true })
  local powered = torch.linalg.matrix_power(a, 3)
  assert_true(powered:requires_grad())
  powered:sum():backward()
  -- d sum(A^3) / dA, matching PyTorch for this matrix.
  assert_near(a:grad():totable(), {
    { 51, 87 },
    { 67, 111 },
  }, 1e-10)
end)

test("matrix_power gradients flow through negative powers and zero", function()
  local a = torch.tensor({ { 4, 7 }, { 2, 6 } },
    { dtype = torch.float64, requires_grad = true })
  a:matrix_power(-1):sum():backward()
  assert_near(a:grad():totable(), {
    { 0.04, -0.08 },
    { -0.03, 0.06 },
  }, 1e-10)

  local b = torch.tensor({ { 1, 2 }, { 3, 4 } },
    { dtype = torch.float64, requires_grad = true })
  local identity = b:matrix_power(0)
  assert_true(identity:requires_grad())
  assert_tensor(identity, { { 1, 0 }, { 0, 1 } })
  identity:sum():backward()
  assert_near(b:grad():totable(), { { 0, 0 }, { 0, 0 } }, 1e-12)
end)

test("trapezoid gradients flow to values and sample positions", function()
  local y = torch.tensor({ 1, 2, 3 },
    { dtype = torch.float64, requires_grad = true })
  local x = torch.tensor({ 0, 1, 3 },
    { dtype = torch.float64, requires_grad = true })
  local integral = torch.trapezoid(y, x)
  assert_true(integral:requires_grad())
  integral:backward()
  assert_near(y:grad():totable(), { 0.5, 1.5, 1 }, 1e-12)
  assert_near(x:grad():totable(), { -1.5, -1, 2.5 }, 1e-12)
end)
