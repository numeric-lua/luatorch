local torch = require("torch")

local function float_matrix(rows)
  return torch.tensor(rows, { dtype = torch.float64 })
end

local function deep_copy(value)
  if type(value) ~= "table" then return value end
  local copy = {}
  for key, item in pairs(value) do
    copy[key] = deep_copy(item)
  end
  return copy
end

test("svdvals returns descending singular values", function()
  local matrix = float_matrix({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local values = matrix:svdvals()
  assert_shape(values, { 2 })
  assert_dtype(values, "float64")
  local _, expected = matrix:svd(false)
  assert_tensor(values, expected:totable(), 1e-12)
  assert_true(values:get(1) >= values:get(2), "singular values are sorted")

  local u, _, vh = matrix:svd(false)
  local diagonal = torch.zeros(2, 2, { dtype = torch.float64 })
  for i = 1, 2 do diagonal:set(i, i, values:get(i)) end
  assert_tensor(u:matmul(diagonal):matmul(vh), matrix:totable(), 1e-10)

  local alias = torch.linalg.svdvals(matrix)
  assert_tensor(alias, values:totable(), 1e-12)
  assert_dtype(torch.linalg.svdvals(matrix:float()), "float32")
end)

test("svdvals handles wide, batched, and empty matrices", function()
  local wide = float_matrix({ { 1, 2, 3, 4 }, { 5, 6, 7, 8 } })
  assert_shape(wide:svdvals(), { 2 })

  local batch = torch.tensor({
    { { 3, 0 }, { 0, 2 } },
    { { 1, 2 }, { 3, 4 } },
  }, { dtype = torch.float64 })
  local values = batch:svdvals()
  assert_shape(values, { 2, 2 })
  assert_tensor(values:select(1, 1), { 3, 2 }, 1e-12)
  for i = 1, 2 do
    local _, expected = batch:select(1, i):svd(false)
    assert_tensor(values:select(1, i), expected:totable(), 1e-10)
  end

  assert_shape(torch.zeros(2, 0, 3, {
    dtype = torch.float64,
  }):svdvals(), { 2, 0 })
  assert_shape(torch.zeros(0, 2, {
    dtype = torch.float64,
  }):svdvals(), { 0 })
end)

test("svdvals gradients match central differences", function()
  local values = {
    { 3.0, 0.4 },
    { 0.2, 2.0 },
    { 0.1, 0.3 },
  }
  local weights = torch.tensor({ 1.3, -0.7 }, { dtype = torch.float64 })
  local matrix = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local loss = (matrix:svdvals() * weights):sum()
  loss:backward()

  local eps = 1e-6
  local function numeric(i, j, delta)
    local copy = { { values[1][1], values[1][2] },
      { values[2][1], values[2][2] }, { values[3][1], values[3][2] } }
    copy[i][j] = copy[i][j] + delta
    local perturbed = torch.tensor(copy, { dtype = torch.float64 })
    return (perturbed:svdvals() * weights):sum():item()
  end
  for i = 1, 3 do
    for j = 1, 2 do
      assert_near(matrix:grad():get(i, j),
        (numeric(i, j, eps) - numeric(i, j, -eps)) / (2 * eps), 1e-5,
        string.format("d svdvals loss / d A[%d][%d]", i, j))
    end
  end
end)

test("svdvals validates ranks and dtypes", function()
  assert_error(function()
    torch.linalg.svdvals(torch.ones(3))
  end, "matrix or a batch")
  assert_error(function()
    torch.linalg.svdvals(torch.ones(2, 2, { dtype = torch.float16 }))
  end, "float32 and float64")
  assert_error(function()
    torch.linalg.svdvals(torch.ones(2, 2, { dtype = torch.int64 }))
  end, "float32 and float64")
  assert_error(function()
    torch.linalg.svdvals(torch.ones(2, 2, { dtype = torch.bool }))
  end, "float32 and float64")
end)

test("vecdot reduces a dimension with broadcasting", function()
  local x = float_matrix({ { 1, 2 }, { 3, 4 } })
  local y = float_matrix({ { 5, 6 }, { 7, 8 } })
  assert_tensor(torch.linalg.vecdot(x, y), { 17, 53 })
  assert_tensor(x:vecdot(y, 1), { 26, 44 })
  assert_tensor(x:vecdot(y, -2), { 26, 44 })
  assert_shape(torch.linalg.vecdot(x, y, 1), { 2 })

  local vector = torch.tensor({ 5, 6 }, { dtype = torch.float64 })
  assert_tensor(torch.linalg.vecdot(x, vector), { 17, 39 })
  assert_tensor(torch.linalg.vecdot(vector, x), { 17, 39 })

  local third = torch.tensor({
    { { 1, 2 }, { 3, 4 } },
    { { 5, 6 }, { 7, 8 } },
  }, { dtype = torch.float64 })
  local weights = float_matrix({ { 2, -1 } })
  assert_shape(third:vecdot(weights, 3), { 2, 2 })
  assert_tensor(third:vecdot(weights, 3), {
    { 0, 2 },
    { 4, 6 },
  })
  assert_tensor(third:vecdot(weights, 2), { { 8, -6 }, { 24, -14 } })

  local first = torch.tensor({ 1, 2 }, { dtype = torch.float64 })
  local second = torch.tensor({ 3, 4 }, { dtype = torch.float64 })
  local scalar = torch.linalg.vecdot(first, second)
  assert_shape(scalar, {})
  assert_near(scalar:item(), 11, 1e-12)
end)

test("vecdot handles empty vectors, dtypes, and gradients", function()
  local empty = torch.zeros(2, 0, { dtype = torch.float64 })
  assert_tensor(empty:vecdot(empty), { 0, 0 })

  local x = torch.ones(2, 3)
  local y = torch.ones(2, 3)
  assert_dtype(x:vecdot(y), "float32")
  assert_dtype(x:half():vecdot(y:half()), "float16")
  assert_dtype(x:double():vecdot(y:double()), "float64")

  local values_x = { { 1.2, -0.4 }, { 0.5, 1.1 } }
  local values_y = { { -0.3, 0.8 }, { 2.0, -1.2 } }
  local left = torch.tensor(values_x, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local right = torch.tensor(values_y, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local loss = torch.linalg.vecdot(left, right):sum()
  loss:backward()

  local eps = 1e-6
  local function numeric(side, i, j, delta)
    local copy = { { values_x[1][1], values_x[1][2] },
      { values_x[2][1], values_x[2][2] } }
    local other = { { values_y[1][1], values_y[1][2] },
      { values_y[2][1], values_y[2][2] } }
    local target = side == "x" and copy or other
    target[i][j] = target[i][j] + delta
    local a = torch.tensor(copy, { dtype = torch.float64 })
    local b = torch.tensor(other, { dtype = torch.float64 })
    return torch.linalg.vecdot(a, b):sum():item()
  end
  for i = 1, 2 do
    for j = 1, 2 do
      assert_near(left:grad():get(i, j),
        (numeric("x", i, j, eps) - numeric("x", i, j, -eps)) / (2 * eps),
        1e-5)
      assert_near(right:grad():get(i, j),
        (numeric("y", i, j, eps) - numeric("y", i, j, -eps)) / (2 * eps),
        1e-5)
    end
  end
end)

test("vecdot validates dtypes, dimensions, and broadcasting", function()
  local x = float_matrix({ { 1, 2 }, { 3, 4 } })
  assert_error(function()
    torch.linalg.vecdot(x, { 1, 2 })
  end, "two tensors")
  assert_error(function()
    x:vecdot(torch.ones(2, 2))
  end, "same dtype")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):vecdot(
      torch.ones(2, 2, { dtype = torch.int64 }))
  end, "float16, float32, and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.bool }):vecdot(
      torch.ones(2, 2, { dtype = torch.bool }))
  end, "float16, float32, and float64")
  assert_error(function()
    torch.linalg.vecdot(torch.tensor(1), torch.tensor(2))
  end, "at least one dimension")
  assert_error(function()
    x:vecdot(x, 3)
  end, "dimension out of range")
  assert_error(function()
    x:vecdot(float_matrix({ { 1, 2, 3 }, { 4, 5, 6 } }))
  end, "size")
end)

test("tensorinv inverts tensordot groupings", function()
  local matrix = float_matrix({ { 4, 7 }, { 2, 6 } })
  assert_tensor(matrix:tensorinv(1), matrix:inverse():totable(), 1e-12)
  assert_tensor(torch.linalg.tensorinv(matrix, 1):
    matmul(matrix), { { 1, 0 }, { 0, 1 } }, 1e-12)

  local grouped = torch.eye(24, { dtype = torch.float64 })
    :reshape(6, 4, 2, 3, 4)
  local inverse = torch.linalg.tensorinv(grouped)
  assert_shape(inverse, { 2, 3, 4, 6, 4 })
  local product = grouped:reshape(24, 24):matmul(inverse:reshape(24, 24))
  assert_tensor(product, torch.eye(24, {
    dtype = torch.float64,
  }):totable(), 1e-10)

  local rectangular = torch.eye(6, { dtype = torch.float64 })
    :reshape(2, 3, 6)
  local right = torch.linalg.tensorinv(rectangular, 2)
  assert_shape(right, { 6, 2, 3 })
  assert_tensor(rectangular:reshape(6, 6):matmul(right:reshape(6, 6)),
    torch.eye(6, { dtype = torch.float64 }):totable(), 1e-10)

  local shaped = torch.tensor({
    { { 2, 0, 0, 0 }, { 0, 4, 0, 0 } },
    { { 0, 0, 1, 0 }, { 0, 0, 0, 3 } },
  }, { dtype = torch.float64 })
  assert_shape(torch.linalg.tensorinv(shaped, 2), { 4, 2, 2 })
  assert_shape(torch.linalg.tensorinv(torch.zeros(0, 0, {
    dtype = torch.float64,
  }), 1), { 0, 0 })
  assert_dtype(torch.linalg.tensorinv(torch.eye(2), 1), "float32")
end)

test("tensorinv gradients match central differences", function()
  local values = {
    { { 2.0, 0.3, 0.1, 0.0 }, { 0.1, 1.5, 0.2, 0.0 } },
    { { 0.2, -0.1, 2.2, 0.4 }, { 0.4, 0.2, 0.1, 1.8 } },
  }
  local tensor = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local weights = torch.tensor({
    { { 1.0, -2.0 }, { 0.5, 3.0 } },
    { { -1.5, 0.25 }, { 2.0, -0.5 } },
    { { 0.75, 1.0 }, { -0.25, 2.5 } },
    { { -1.0, 0.5 }, { 1.5, -3.0 } },
  }, { dtype = torch.float64 })
  local loss = (tensor:tensorinv(2) * weights):sum()
  loss:backward()

  local eps = 1e-6
  local function numeric(i, j, k, delta)
    local copy = {
      { { values[1][1][1], values[1][1][2], values[1][1][3],
        values[1][1][4] },
        { values[1][2][1], values[1][2][2], values[1][2][3],
          values[1][2][4] } },
      { { values[2][1][1], values[2][1][2], values[2][1][3],
        values[2][1][4] },
        { values[2][2][1], values[2][2][2], values[2][2][3],
          values[2][2][4] } },
    }
    copy[i][j][k] = copy[i][j][k] + delta
    local perturbed = torch.tensor(copy, { dtype = torch.float64 })
    return (perturbed:tensorinv(2) * weights):sum():item()
  end
  for i = 1, 2 do
    for j = 1, 2 do
      for k = 1, 4 do
        assert_near(tensor:grad():get(i, j, k),
          (numeric(i, j, k, eps) - numeric(i, j, k, -eps)) / (2 * eps),
          1e-5)
      end
    end
  end
end)

test("tensorinv validates ind, shape products, and dtypes", function()
  local matrix = float_matrix({ { 1, 2 }, { 3, 4 } })
  assert_error(function()
    torch.linalg.tensorinv(torch.tensor(1, { dtype = torch.float64 }))
  end, "at least one dimension")
  assert_error(function()
    matrix:tensorinv(0)
  end, "in 1..2")
  assert_error(function()
    matrix:tensorinv(3)
  end, "in 1..2")
  assert_error(function()
    matrix:tensorinv(1.5)
  end, "integer")
  assert_error(function()
    torch.ones(2, 3, { dtype = torch.float64 }):tensorinv(1)
  end, "got 2 and 3")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float16 }):tensorinv(1)
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):tensorinv(1)
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.bool }):tensorinv(1)
  end, "float32 and float64")
end)

test("tensorsolve solves grouped tensor systems", function()
  local identity = torch.eye(24, { dtype = torch.float64 })
  local a = identity:reshape(6, 4, 2, 3, 4)
  local b = torch.tensor({
    { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 },
    { 13, 14, 15, 16 }, { 17, 18, 19, 20 }, { 21, 22, 23, 24 },
  }, { dtype = torch.float64 })
  local x = torch.linalg.tensorsolve(a, b)
  assert_shape(x, { 2, 3, 4 })
  assert_tensor(a:reshape(24, 24):matmul(x:reshape(24)),
    b:reshape(24):totable(), 1e-12)

  local matrix = float_matrix({ { 3, 1 }, { 1, 2 } })
  local rhs = torch.tensor({ 9, 8 }, { dtype = torch.float64 })
  assert_tensor(torch.linalg.tensorsolve(matrix, rhs),
    matrix:solve(rhs):totable(), 1e-12)

  -- Move dims 1 and 3 to the end, then solve the 24x24 grouped system.
  local moved = identity:reshape(4, 3, 2, 6, 4)
  local original = moved:permute(4, 1, 5, 2, 3)
  local expected = torch.arange(1, 25, 1, {
    dtype = torch.float64,
  }):reshape(6, 4)
  local target = identity:matmul(expected:reshape(24)):reshape(4, 3, 2)
  local solved = torch.linalg.tensorsolve(original, target, { 1, 3 })
  assert_shape(solved, { 6, 4 })
  assert_tensor(solved, expected:totable(), 1e-10)

  local batch = float_matrix({
    { 2, 1, 0, 0 },
    { 1, 3, 1, 0 },
    { 0, 1, 2, 1 },
    { 0, 0, 1, 3 },
  }):reshape(2, 2, 2, 2)
  local batch_expected = float_matrix({ { 1, 3 }, { 2, 2 } })
  local batch_rhs = batch:reshape(4, 4):matmul(
    batch_expected:reshape(4)):reshape(2, 2)
  local batch_x = torch.linalg.tensorsolve(batch, batch_rhs)
  assert_shape(batch_x, { 2, 2 })
  assert_tensor(batch_x, batch_expected:totable(), 1e-12)

  assert_shape(torch.linalg.tensorsolve(torch.zeros(0, 0, {
    dtype = torch.float64,
  }), torch.zeros(0, { dtype = torch.float64 })), { 0 })
end)

test("tensorsolve gradients match central differences", function()
  local values_a = {
    { { { 2.0, 0.3 }, { 0.1, 1.5 } },
      { { 0.2, -0.1 }, { 0.4, 2.2 } } },
    { { { 0.5, 0.2 }, { -0.3, 1.1 } },
      { { 0.6, 0.0 }, { 0.2, 1.9 } } },
  }
  local values_b = { { 1.0, -0.5 }, { 2.0, 3.0 } }
  local a = torch.tensor(values_a, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local b = torch.tensor(values_b, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local weights = float_matrix({ { 1.0, -2.0 }, { 0.5, 3.0 } })
  local loss = (a:tensorsolve(b) * weights):sum()
  loss:backward()

  local eps = 1e-6
  local function numeric(side, i, j, k, l, delta)
    local copy_a = deep_copy(values_a)
    local copy_b = deep_copy(values_b)
    if side == "a" then
      copy_a[i][j][k][l] = copy_a[i][j][k][l] + delta
    else
      copy_b[i][j] = copy_b[i][j] + delta
    end
    local ta = torch.tensor(copy_a, { dtype = torch.float64 })
    local tb = torch.tensor(copy_b, { dtype = torch.float64 })
    return (ta:tensorsolve(tb) * weights):sum():item()
  end
  for i = 1, 2 do
    for j = 1, 2 do
      for k = 1, 2 do
        for l = 1, 2 do
          assert_near(a:grad():get(i, j, k, l),
            (numeric("a", i, j, k, l, eps) -
              numeric("a", i, j, k, l, -eps)) / (2 * eps), 1e-5)
        end
      end
    end
  end
  for i = 1, 2 do
    for j = 1, 2 do
      assert_near(b:grad():get(i, j),
        (numeric("b", i, j, 1, 1, eps) -
          numeric("b", i, j, 1, 1, -eps)) / (2 * eps), 1e-5)
    end
  end
end)

test("tensorsolve validates shapes, dims, and dtypes", function()
  local a = float_matrix({ { 1, 2 }, { 3, 4 } })
  local b = torch.tensor({ 1, 2 }, { dtype = torch.float64 })
  assert_error(function()
    a:tensorsolve({ 1, 2 })
  end, "right-hand side")
  assert_error(function()
    torch.linalg.tensorsolve(a, torch.ones(2, 2, 2, {
      dtype = torch.float64,
    }))
  end, "no more")
  assert_error(function()
    a:tensorsolve(b, 1)
  end, "table of dimensions")
  assert_error(function()
    a:tensorsolve(b, { 1, 1 })
  end, "unique")
  assert_error(function()
    a:tensorsolve(b, { 3 })
  end, "out of range")
  assert_error(function()
    torch.ones(2, 3, { dtype = torch.float64 }):tensorsolve(
      torch.ones(2, { dtype = torch.float64 }))
  end, "got 2 and 3")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float64 }):tensorsolve(
      torch.ones(3, { dtype = torch.float64 }))
  end, "have 2 elements")
  assert_error(function()
    a:tensorsolve(torch.ones(2, { dtype = torch.float32 }))
  end, "match the tensor dtype")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float16 }):tensorsolve(
      torch.ones(2, { dtype = torch.float16 }))
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):tensorsolve(
      torch.ones(2, { dtype = torch.int64 }))
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.bool }):tensorsolve(
      torch.ones(2, { dtype = torch.bool }))
  end, "float32 and float64")
  assert_error(function()
    torch.linalg.tensorsolve(torch.tensor(1.0, {
      dtype = torch.float64,
    }), torch.tensor(1.0, { dtype = torch.float64 }))
  end, "at least one dimension")
end)

test("householder_product reconstructs reflectors", function()
  local column = float_matrix({ { 9 }, { 3 }, { 4 } })
  local tau = torch.tensor({ 2 / 26 }, { dtype = torch.float64 })
  assert_tensor(torch.linalg.householder_product(column, tau), {
    { 12 / 13 },
    { -3 / 13 },
    { -4 / 13 },
  }, 1e-12)

  local a = float_matrix({
    { 7, 8, 9 },
    { 0.6, 9, 1 },
    { 0.8, 0.2, 5 },
  })
  local values = torch.tensor({
    2 / (0.6 ^ 2 + 0.8 ^ 2),
    2 / (1 + 0.2 ^ 2),
    0.5,
  }, { dtype = torch.float64 })
  local q = torch.linalg.householder_product(a, values)
  assert_shape(q, { 3, 3 })

  local identity = torch.eye(3, { dtype = torch.float64 })
  local b1 = torch.tensor({ 1, 0.6, 0.8 }, { dtype = torch.float64 })
  local b2 = torch.tensor({ 0, 1, 0.2 }, { dtype = torch.float64 })
  local b3 = torch.tensor({ 0, 0, 1 }, { dtype = torch.float64 })
  local h1 = identity - values:get(1) * b1:outer(b1)
  local h2 = identity - values:get(2) * b2:outer(b2)
  local h3 = identity - values:get(3) * b3:outer(b3)
  assert_tensor(q, h1:matmul(h2):matmul(h3):totable(), 1e-12)
  assert_dtype(torch.linalg.householder_product(a:float(), values:float()),
    "float32")
end)

test("householder_product handles batches, partial tau, and empties", function()
  local a = float_matrix({
    { 7, 8, 9 },
    { 0.6, 9, 1 },
    { 0.8, 0.2, 5 },
  })
  local tau = torch.tensor({
    2 / (0.6 ^ 2 + 0.8 ^ 2),
    2 / (1 + 0.2 ^ 2),
    0.5,
  }, { dtype = torch.float64 })
  local single = a:householder_product(tau)
  local batch_a = a:unsqueeze(1):expand(2, 3, 3):contiguous()
  local batch_tau = tau:unsqueeze(1):expand(2, 3):contiguous()
  local batch = batch_a:householder_product(batch_tau)
  assert_shape(batch, { 2, 3, 3 })
  for i = 1, 2 do
    assert_tensor(batch:select(1, i), single:totable(), 1e-12)
  end

  local tall = float_matrix({
    { 7, 8 },
    { 0.6, 9 },
    { 0.8, 0.2 },
    { 0.4, 0.1 },
  })
  local one = torch.tensor({ 2 / (0.6 ^ 2 + 0.8 ^ 2 + 0.4 ^ 2) },
    { dtype = torch.float64 })
  local q = tall:householder_product(one)
  assert_shape(q, { 4, 2 })
  local b = torch.tensor({ 1, 0.6, 0.8, 0.4 }, { dtype = torch.float64 })
  local h = torch.eye(4, { dtype = torch.float64 }) -
    one:get(1) * b:outer(b)
  assert_tensor(q, h:narrow(2, 1, 2):totable(), 1e-12)

  assert_shape(torch.zeros(3, 0, { dtype = torch.float64 })
    :householder_product(torch.zeros(0, { dtype = torch.float64 })), { 3, 0 })
  assert_shape(torch.zeros(0, 0, { dtype = torch.float64 })
    :householder_product(torch.zeros(0, { dtype = torch.float64 })), { 0, 0 })
end)

test("householder_product gradients match central differences", function()
  local values_a = {
    { 7, 8, 9 },
    { 0.6, 9, 1 },
    { 0.8, 0.2, 5 },
  }
  local values_tau = {
    2 / (0.6 ^ 2 + 0.8 ^ 2),
    2 / (1 + 0.2 ^ 2),
    0.5,
  }
  local a = torch.tensor(values_a, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local tau = torch.tensor(values_tau, {
    dtype = torch.float64,
    requires_grad = true,
  })
  a:householder_product(tau):sum():backward()

  local eps = 1e-6
  local function numeric(side, i, j, delta)
    local copy_a = deep_copy(values_a)
    local copy_tau = { values_tau[1], values_tau[2], values_tau[3] }
    if side == "a" then
      copy_a[i][j] = copy_a[i][j] + delta
    else
      copy_tau[i] = copy_tau[i] + delta
    end
    local ta = torch.tensor(copy_a, { dtype = torch.float64 })
    local tt = torch.tensor(copy_tau, { dtype = torch.float64 })
    return ta:householder_product(tt):sum():item()
  end
  for i = 1, 3 do
    for j = 1, 3 do
      assert_near(a:grad():get(i, j),
        (numeric("a", i, j, eps) - numeric("a", i, j, -eps)) / (2 * eps),
        1e-5)
    end
  end
  for i = 1, 3 do
    assert_near(tau:grad():get(i),
      (numeric("tau", i, 1, eps) - numeric("tau", i, 1, -eps)) / (2 * eps),
      1e-5)
  end
end)

test("householder_product validates shapes, tau, and dtypes", function()
  local a = float_matrix({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local tau = torch.tensor({ 0.5, 0.25 }, { dtype = torch.float64 })
  assert_error(function()
    a:householder_product({ 0.5, 0.25 })
  end, "tau tensor")
  assert_error(function()
    torch.linalg.householder_product(torch.ones(2), torch.ones(1, {
      dtype = torch.float64,
    }))
  end, "matrix or a batch")
  assert_error(function()
    torch.ones(3, 2, { dtype = torch.float16 }):householder_product(
      torch.ones(2, { dtype = torch.float16 }))
  end, "float32 and float64")
  assert_error(function()
    torch.ones(3, 2, { dtype = torch.int64 }):householder_product(
      torch.ones(2, { dtype = torch.int64 }))
  end, "float32 and float64")
  assert_error(function()
    torch.ones(3, 2, { dtype = torch.bool }):householder_product(
      torch.ones(2, { dtype = torch.bool }))
  end, "float32 and float64")
  assert_error(function()
    a:householder_product(tau:float())
  end, "match the input dtype")
  assert_error(function()
    float_matrix({ { 1, 2, 3 }, { 4, 5, 6 } }):householder_product(
      torch.ones(2, { dtype = torch.float64 }))
  end, "input.shape[-2] >= input.shape[-1]")
  assert_error(function()
    a:householder_product(torch.ones(1, 2, { dtype = torch.float64 }))
  end, "one dimension less")
  assert_error(function()
    torch.ones(2, 3, 2, { dtype = torch.float64 }):householder_product(
      torch.ones(1, 2, { dtype = torch.float64 }))
  end, "batch dimensions")
  assert_error(function()
    a:householder_product(torch.ones(3, { dtype = torch.float64 }))
  end, "tau:size(-1)")
end)
