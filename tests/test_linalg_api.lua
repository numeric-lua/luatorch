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

test("eigh returns ascending eigenvalues and orthonormal eigenvectors", function()
  local matrix = float_matrix({ { 2, 1 }, { 1, 3 } })
  local values, vectors = torch.linalg.eigh(matrix)
  assert_shape(values, { 2 })
  assert_shape(vectors, { 2, 2 })
  assert_dtype(values, "float64")
  assert_dtype(vectors, "float64")
  assert_true(values:get(1) < values:get(2), "eigenvalues are ascending")
  assert_near(values:get(1), (5 - math.sqrt(5)) / 2, 1e-12)
  assert_near(values:get(2), (5 + math.sqrt(5)) / 2, 1e-12)

  local product = vectors:t():matmul(vectors)
  assert_tensor(product, torch.eye(2, { dtype = torch.float64 }):totable(),
    1e-12)
  local diagonal = torch.zeros(2, 2, { dtype = torch.float64 })
  for i = 1, 2 do diagonal:set(i, i, values:get(i)) end
  assert_tensor(vectors:matmul(diagonal):matmul(vectors:t()),
    matrix:totable(), 1e-12)

  local upper_values, upper_vectors = torch.linalg.eigh(matrix, "U")
  assert_tensor(upper_values, values:totable(), 1e-12)
  assert_tensor(upper_vectors, vectors:totable(), 1e-12)
  local lower_values, lower_vectors = torch.linalg.eigh(matrix, "u")
  assert_tensor(lower_values, values:totable(), 1e-12)
  assert_tensor(lower_vectors, vectors:totable(), 1e-12)
  assert_dtype(torch.linalg.eigh(matrix:float()), "float32")
end)

test("eigh handles batches and empty matrices", function()
  local batch = torch.tensor({
    { { 3, 0 }, { 0, 2 } },
    { { 1, 2 }, { 2, 1 } },
  }, { dtype = torch.float64 })
  local values, vectors = batch:eigh()
  assert_shape(values, { 2, 2 })
  assert_shape(vectors, { 2, 2, 2 })
  assert_tensor(values:select(1, 1), { 2, 3 }, 1e-12)
  assert_tensor(values:select(1, 2), { -1, 3 }, 1e-12)
  for i = 1, 2 do
    local single_values, single_vectors = batch:select(1, i):eigh()
    assert_tensor(values:select(1, i), single_values:totable(), 1e-12)
    assert_tensor(vectors:select(1, i), single_vectors:totable(), 1e-12)
  end

  local empty_values, empty_vectors = torch.zeros(2, 0, 0, {
    dtype = torch.float64,
  }):eigh()
  assert_shape(empty_values, { 2, 0 })
  assert_shape(empty_vectors, { 2, 0, 0 })
  assert_shape(torch.zeros(0, 0, {
    dtype = torch.float64,
  }):eigvalsh(), { 0 })
end)

test("eigvalsh matches eigh and differentiates analytically", function()
  local values_a = { { 2.0, 0.4 }, { 0.4, 1.5 } }
  local matrix = torch.tensor(values_a, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local eigenvalues = torch.linalg.eigvalsh(matrix)
  local expected, vectors = torch.linalg.eigh(matrix)
  assert_tensor(eigenvalues, expected:totable(), 1e-12)
  assert_dtype(torch.linalg.eigvalsh(matrix:float()), "float32")

  local weights = torch.tensor({ 0.75, -1.25 }, { dtype = torch.float64 })
  local loss = (eigenvalues * weights):sum()
  loss:backward()
  local diagonal = torch.zeros(2, 2, { dtype = torch.float64 })
  for i = 1, 2 do diagonal:set(i, i, weights:get(i)) end
  local analytic = vectors:matmul(diagonal):matmul(vectors:t())
  assert_tensor(matrix:grad(), analytic:totable(), 1e-10)
end)

test("eigh validates ranks, square inputs, UPLO, and dtypes", function()
  assert_error(function()
    torch.linalg.eigh(torch.ones(3))
  end, "matrix or a batch")
  assert_error(function()
    torch.linalg.eigh(torch.ones(2, 3, 2, { dtype = torch.float64 }))
  end, "square")
  assert_error(function()
    torch.linalg.eigh(torch.ones(2, 2, { dtype = torch.float64 }), "X")
  end, "UPLO")
  assert_error(function()
    torch.linalg.eigh(torch.ones(2, 2, { dtype = torch.float64 }), 1)
  end, "UPLO")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float16 }):eigh()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):eigh()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.bool }):eigh()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float64 }):eigvalsh("X")
  end, "UPLO")
  assert_error(function()
    torch.ones(3, 2, { dtype = torch.float64 }):eigvalsh()
  end, "square")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):eigvalsh()
  end, "float32 and float64")
end)

test("cholesky_ex factors matrices and returns info", function()
  local matrix = float_matrix({ { 4, 2 }, { 2, 3 } })
  local factor, info = torch.linalg.cholesky_ex(matrix)
  assert_shape(factor, { 2, 2 })
  assert_dtype(factor, "float64")
  assert_dtype(info, "int32")
  assert_eq(info:item(), 0)
  assert_tensor(factor:matmul(factor:t()), matrix:totable(), 1e-12)
  assert_tensor(factor, matrix:cholesky():totable(), 1e-12)

  local upper, upper_info = matrix:cholesky_ex({ upper = true })
  assert_eq(upper_info:item(), 0)
  assert_tensor(upper:t():matmul(upper), matrix:totable(), 1e-12)
  assert_tensor(upper, matrix:cholesky(true):totable(), 1e-12)

  local single = torch.linalg.cholesky_ex(torch.eye(2):reshape(1, 2, 2))
  assert_shape(single, { 1, 2, 2 })
  assert_dtype(torch.linalg.cholesky_ex(matrix:float()), "float32")
  local empty, empty_info = torch.zeros(0, 0, {
    dtype = torch.float64,
  }):cholesky_ex()
  assert_shape(empty, { 0, 0 })
  assert_eq(empty_info:item(), 0)
end)

test("cholesky_ex reports non-positive-definite matrices", function()
  local batch = torch.tensor({
    { { 4, 2 }, { 2, 3 } },
    { { 1, 2 }, { 2, 1 } },
  }, { dtype = torch.float64 })
  local _, info = torch.linalg.cholesky_ex(batch)
  assert_shape(info, { 2 })
  assert_eq(info:get(1), 0)
  assert_true(info:get(2) > 0, "second matrix should fail the factorization")

  local indefinite = float_matrix({ { 1, 2 }, { 2, 1 } })
  local _, failed_info = indefinite:cholesky_ex()
  assert_true(failed_info:item() > 0)
  assert_error(function()
    indefinite:cholesky_ex({ check_errors = true })
  end, "positive-definite")
end)

test("cholesky_ex differentiates through the factor", function()
  local values = { { 4.0, 1.0 }, { 1.0, 3.0 } }
  local matrix = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local factor = matrix:cholesky_ex()
  local loss = factor:sum()
  loss:backward()

  local eps = 1e-6
  local function numeric(i, j, delta)
    local copy = { { values[1][1], values[1][2] },
      { values[2][1], values[2][2] } }
    copy[i][j] = copy[i][j] + delta
    if i ~= j then copy[j][i] = copy[j][i] + delta end
    return torch.tensor(copy, { dtype = torch.float64 }):cholesky():sum():item()
  end
  -- ATen treats the input as symmetric and returns a symmetric gradient, so
  -- compare the directional derivative along a symmetric perturbation.
  for i = 1, 2 do
    for j = 1, i do
      local actual = matrix:grad():get(i, j)
      if i ~= j then actual = actual + matrix:grad():get(j, i) end
      assert_near(actual,
        (numeric(i, j, eps) - numeric(i, j, -eps)) / (2 * eps), 1e-5)
    end
  end
end)

test("cholesky_ex validates options, shapes, and dtypes", function()
  local matrix = float_matrix({ { 4, 2 }, { 2, 3 } })
  assert_error(function()
    matrix:cholesky_ex({ upper = 1 })
  end, "must be a boolean")
  assert_error(function()
    matrix:cholesky_ex({ check_errors = "yes" })
  end, "must be a boolean")
  assert_error(function()
    matrix:cholesky_ex({ pivot = true })
  end, "unknown option")
  assert_error(function()
    matrix:cholesky_ex(1)
  end, "options must be a table")
  assert_error(function()
    torch.ones(3):cholesky_ex()
  end, "matrix or a batch")
  assert_error(function()
    torch.ones(2, 3, { dtype = torch.float64 }):cholesky_ex()
  end, "square")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float16 }):cholesky_ex()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):cholesky_ex()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.bool }):cholesky_ex()
  end, "float32 and float64")
end)

test("cholesky_inverse recovers the inverse from a Cholesky factor", function()
  local matrix = float_matrix({ { 4, 2 }, { 2, 3 } })
  local expected = matrix:inverse()
  local lower = matrix:cholesky()
  local inverse = lower:cholesky_inverse()
  assert_shape(inverse, { 2, 2 })
  assert_dtype(inverse, "float64")
  assert_tensor(inverse, expected:totable(), 1e-12)
  assert_tensor(matrix:matmul(inverse),
    torch.eye(2, { dtype = torch.float64 }):totable(), 1e-12)
  assert_tensor(lower:t():matmul(inverse):matmul(lower),
    torch.eye(2, { dtype = torch.float64 }):totable(), 1e-12)

  local upper = matrix:cholesky(true)
  local upper_inverse = upper:cholesky_inverse(true)
  assert_tensor(upper_inverse, expected:totable(), 1e-12)
  assert_tensor(upper:matmul(upper_inverse):matmul(upper:t()),
    torch.eye(2, { dtype = torch.float64 }):totable(), 1e-12)

  assert_tensor(torch.linalg.cholesky_inverse(lower), inverse:totable(), 1e-12)
  assert_tensor(torch.cholesky_inverse(upper, true), inverse:totable(), 1e-12)
  assert_dtype(lower:float():cholesky_inverse(), "float32")
end)

test("cholesky_inverse handles batches and gradients", function()
  local batch = torch.tensor({
    { { 4, 2 }, { 2, 3 } },
    { { 2, 1 }, { 1, 2 } },
  }, { dtype = torch.float64 })
  local factors = batch:cholesky()
  local inverse = factors:cholesky_inverse()
  assert_shape(inverse, { 2, 2, 2 })
  for i = 1, 2 do
    assert_tensor(inverse:select(1, i),
      batch:select(1, i):inverse():totable(), 1e-12)
  end

  local values = { { 2, 0 }, { 1, 1.5 } }
  local weights = float_matrix({ { 1.5, -0.5 }, { 2, 0.75 } })
  local factor = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local loss = (factor:cholesky_inverse() * weights):sum()
  loss:backward()

  local eps = 1e-6
  local function numeric(i, j, delta)
    local copy = { { values[1][1], values[1][2] },
      { values[2][1], values[2][2] } }
    copy[i][j] = copy[i][j] + delta
    local perturbed = torch.tensor(copy, { dtype = torch.float64 })
    return (perturbed:cholesky_inverse() * weights):sum():item()
  end
  for i = 1, 2 do
    for j = 1, i do
      assert_near(factor:grad():get(i, j),
        (numeric(i, j, eps) - numeric(i, j, -eps)) / (2 * eps), 1e-5,
        string.format("d cholesky_inverse loss / d L[%d][%d]", i, j))
    end
  end
end)

test("cholesky_inverse validates shapes, dtypes, and upper", function()
  local matrix = float_matrix({ { 4, 2 }, { 2, 3 } })
  assert_error(function()
    torch.linalg.cholesky_inverse(torch.ones(3))
  end, "matrix or a batch")
  assert_error(function()
    torch.linalg.cholesky_inverse(torch.ones(2, 3, { dtype = torch.float64 }))
  end, "square matrices")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float16 }):cholesky_inverse()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):cholesky_inverse()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.bool }):cholesky_inverse()
  end, "float32 and float64")
  assert_error(function()
    matrix:cholesky_inverse(1)
  end, "upper must be a boolean")
  assert_error(function()
    torch.cholesky_inverse({ 1, 2 })
  end, "expects a tensor")
end)

test("inv_ex inverts matrices and returns info", function()
  local matrix = float_matrix({ { 4, 2 }, { 2, 3 } })
  local inverse, info = torch.linalg.inv_ex(matrix)
  assert_shape(inverse, { 2, 2 })
  assert_dtype(inverse, "float64")
  assert_dtype(info, "int32")
  assert_eq(info:item(), 0)
  assert_tensor(matrix:matmul(inverse),
    torch.eye(2, { dtype = torch.float64 }):totable(), 1e-12)
  assert_tensor(inverse, matrix:inverse():totable(), 1e-12)

  local batch = torch.tensor({
    { { 4, 2 }, { 2, 3 } },
    { { 2, 0 }, { 0, 5 } },
  }, { dtype = torch.float64 })
  local batch_inverse, batch_info = batch:inv_ex()
  assert_shape(batch_inverse, { 2, 2, 2 })
  assert_shape(batch_info, { 2 })
  assert_eq(batch_info:get(1), 0)
  assert_eq(batch_info:get(2), 0)
  for i = 1, 2 do
    assert_tensor(batch:select(1, i):matmul(batch_inverse:select(1, i)),
      torch.eye(2, { dtype = torch.float64 }):totable(), 1e-12)
  end

  assert_dtype(torch.linalg.inv_ex(matrix:float()), "float32")
  local empty, empty_info = torch.zeros(0, 0, {
    dtype = torch.float64,
  }):inv_ex()
  assert_shape(empty, { 0, 0 })
  assert_eq(empty_info:item(), 0)
end)

test("inv_ex reports singular matrices", function()
  local batch = torch.tensor({
    { { 4, 2 }, { 2, 3 } },
    { { 1, 2 }, { 2, 4 } },
  }, { dtype = torch.float64 })
  local _, info = torch.linalg.inv_ex(batch)
  assert_shape(info, { 2 })
  assert_eq(info:get(1), 0)
  assert_true(info:get(2) > 0, "second matrix should be singular")

  local singular = float_matrix({ { 1, 2 }, { 2, 4 } })
  local _, singular_info = singular:inv_ex()
  assert_true(singular_info:item() > 0)
  assert_error(function()
    singular:inv_ex({ check_errors = true })
  end, "singular")
end)

test("inv_ex differentiates through the inverse", function()
  local values = { { 4.0, 1.0 }, { 1.0, 3.0 } }
  local matrix = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local inverse = matrix:inv_ex()
  local loss = inverse:sum()
  loss:backward()

  local eps = 1e-6
  local function numeric(i, j, delta)
    local copy = { { values[1][1], values[1][2] },
      { values[2][1], values[2][2] } }
    copy[i][j] = copy[i][j] + delta
    return torch.tensor(copy, { dtype = torch.float64 }):inverse():sum():item()
  end
  for i = 1, 2 do
    for j = 1, 2 do
      assert_near(matrix:grad():get(i, j),
        (numeric(i, j, eps) - numeric(i, j, -eps)) / (2 * eps), 1e-5)
    end
  end
end)

test("inv_ex validates options, shapes, and dtypes", function()
  local matrix = float_matrix({ { 4, 2 }, { 2, 3 } })
  assert_error(function()
    matrix:inv_ex({ check_errors = 1 })
  end, "must be a boolean")
  assert_error(function()
    matrix:inv_ex({ upper = true })
  end, "unknown option")
  assert_error(function()
    matrix:inv_ex("yes")
  end, "options must be a table")
  assert_error(function()
    torch.ones(3):inv_ex()
  end, "matrix or a batch")
  assert_error(function()
    torch.ones(2, 3, { dtype = torch.float64 }):inv_ex()
  end, "square")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float16 }):inv_ex()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):inv_ex()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.bool }):inv_ex()
  end, "float32 and float64")
end)

test("matrix_norm handles batches and all orders", function()
  local batch = torch.tensor({
    { { 1, 2 }, { 3, 4 } },
    { { 2, 0 }, { 0, 1 } },
  }, { dtype = torch.float64 })
  local expected = {
    { "fro", { math.sqrt(30), math.sqrt(5) } },
    { "nuc", { 5.830951894845301, 3 } },
    { 1, { 6, 2 } },
    { -1, { 4, 1 } },
    { 2, { 5.464985704219043, 2 } },
    { -2, { 0.3659661906262576, 1 } },
    { math.huge, { 7, 2 } },
    { -math.huge, { 3, 1 } },
  }
  for _, entry in ipairs(expected) do
    local norm = torch.linalg.matrix_norm(batch, entry[1])
    assert_shape(norm, { 2 })
    assert_tensor(norm, entry[2], 1e-10)
    for i = 1, 2 do
      assert_near(norm:get(i), torch.linalg.matrix_norm(
        batch:select(1, i), entry[1]))
    end
  end

  local empty = torch.linalg.matrix_norm(torch.zeros(0, 2, 3, {
    dtype = torch.float64,
  }), "nuc")
  assert_shape(empty, { 0 })
end)

test("matrix_norm reduces arbitrary dimension pairs and keepdims", function()
  local tensor = torch.arange(1, 25, 1, {
    dtype = torch.float64,
  }):reshape(2, 3, 4)
  assert_tensor(torch.linalg.matrix_norm(tensor, "fro", 1, 3), {
    29.597297173897484, 39.344631145812, 49.759421218498915,
  }, 1e-6)
  assert_tensor(torch.linalg.matrix_norm(tensor, 2, 1, 3), {
    29.541494486236118, 39.32095234720612, 49.74772644714978,
  }, 1e-6)
  assert_tensor(torch.linalg.matrix_norm(tensor, "nuc", 1, 3), {
    31.35811319132562, 40.68576241045496, 50.826481905793855,
  }, 1e-6)

  local kept = torch.linalg.matrix_norm(tensor, 1, 1, 3, true)
  assert_shape(kept, { 1, 3, 1 })
  assert_tensor(kept, { { { 20 }, { 28 }, { 36 } } }, 1e-12)
  assert_shape(torch.linalg.matrix_norm(tensor, "fro", 2, 3, true),
    { 2, 1, 1 })
  assert_shape(torch.linalg.matrix_norm(tensor, "fro", 3, 1, true),
    { 1, 3, 1 })
end)

test("matrix_norm gradients match central differences", function()
  local values = {
    { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 },
      { 9.0, 10.0, 11.0, 12.0 } },
    { { 2.0, 1.0, 0.5, 1.5 }, { 0.0, 3.0, 1.0, 2.0 },
      { 1.0, 0.0, 2.5, 4.0 } },
  }
  local tensor = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local loss = torch.linalg.matrix_norm(tensor, "fro", 1, 3):sum() +
    torch.linalg.matrix_norm(tensor, 2, 1, 3):sum()
  loss:backward()

  local eps = 1e-6
  local function numeric(i, j, k, delta)
    local copy = deep_copy(values)
    copy[i][j][k] = copy[i][j][k] + delta
    local perturbed = torch.tensor(copy, { dtype = torch.float64 })
    return (torch.linalg.matrix_norm(perturbed, "fro", 1, 3):sum() +
      torch.linalg.matrix_norm(perturbed, 2, 1, 3):sum()):item()
  end
  for i = 1, 2 do
    for j = 1, 3 do
      for k = 1, 4 do
        assert_near(tensor:grad():get(i, j, k),
          (numeric(i, j, k, eps) - numeric(i, j, k, -eps)) / (2 * eps),
          1e-5)
      end
    end
  end
end)

test("matrix_norm preserves legacy results and validates inputs", function()
  local matrix = float_matrix({ { 1, 2 }, { 3, 4 } })
  assert_eq(type(torch.linalg.matrix_norm(matrix, "fro")), "number")
  assert_near(torch.linalg.matrix_norm(matrix, "fro"), math.sqrt(30), 1e-12)
  assert_shape(torch.linalg.matrix_norm(matrix, "fro", 1, 2, true), { 1, 1 })

  assert_error(function()
    torch.linalg.matrix_norm(torch.ones(3), "fro")
  end, "at least 2 dimensions")
  assert_error(function()
    torch.linalg.matrix_norm(matrix, "fro", 1, 1)
  end, "dims must be different")
  assert_error(function()
    torch.linalg.matrix_norm(matrix, 3)
  end, "matrix_norm supports")
  assert_error(function()
    torch.linalg.matrix_norm(matrix, "fro", 1, 3)
  end, "out of range")
  assert_error(function()
    torch.linalg.matrix_norm(matrix, "fro", 1.5, 2)
  end, "integer")
  assert_error(function()
    torch.linalg.matrix_norm(torch.ones(2, 2, { dtype = torch.float16 }), 2)
  end, "spectral and nuclear")
  assert_error(function()
    torch.linalg.matrix_norm(torch.ones(2, 2, { dtype = torch.int64 }), "nuc")
  end, "spectral and nuclear")
end)

test("torch.linalg.norm dispatches vectors and matrices", function()
  local matrix = torch.arange(1, 7, 1, {
    dtype = torch.float64,
  }):reshape(2, 3)
  assert_near(torch.linalg.norm(matrix):item(), math.sqrt(91), 1e-12)
  assert_near(torch.linalg.norm(matrix, 2):item(),
    torch.linalg.matrix_norm(matrix, 2), 1e-12)
  assert_near(torch.linalg.norm(matrix, 1):item(), 9, 1e-12)
  assert_near(torch.linalg.norm(matrix, -1):item(), 5, 1e-12)
  assert_near(torch.linalg.norm(matrix, math.huge):item(), 15, 1e-12)
  assert_near(torch.linalg.norm(matrix, -math.huge):item(), 6, 1e-12)
  assert_near(torch.linalg.norm(matrix, "fro"):item(), math.sqrt(91), 1e-12)
  assert_near(torch.linalg.norm(matrix, "nuc"):item(),
    torch.linalg.matrix_norm(matrix, "nuc"), 1e-12)

  assert_tensor(torch.linalg.norm(matrix, nil, 1), {
    math.sqrt(17), math.sqrt(29), math.sqrt(45),
  }, 1e-12)
  assert_tensor(torch.linalg.norm(matrix, nil, 2), {
    math.sqrt(14), math.sqrt(77),
  }, 1e-12)
  assert_near(torch.linalg.norm(matrix, 1, { 1, 2 }):item(), 9, 1e-12)
  assert_near(torch.linalg.norm(matrix, 2, { 1, 2 }):item(),
    torch.linalg.matrix_norm(matrix, 2), 1e-12)
  assert_near(torch.linalg.norm(matrix, "fro", { -2, -1 }):item(),
    math.sqrt(91), 1e-12)
  assert_tensor(torch.linalg.norm(matrix, nil, { 2 }), {
    math.sqrt(14), math.sqrt(77),
  }, 1e-12)

  assert_shape(torch.linalg.norm(matrix, "fro", { 1, 2 }, true), { 1, 1 })
  assert_shape(torch.linalg.norm(matrix, nil, 1, true), { 1, 3 })
  assert_shape(torch.linalg.norm(matrix, nil, nil, true), { 1, 1 })
  assert_shape(torch.linalg.norm(torch.ones(2, 3, 4), nil, nil, true),
    { 1, 1, 1 })
  assert_near(torch.linalg.norm(torch.tensor({ 3, 4 }, {
    dtype = torch.float64,
  }), 2):item(), 5, 1e-12)
  assert_near(torch.linalg.norm(torch.tensor({ { 3, 4 } }, {
    dtype = torch.float64,
  }), "fro"):item(), 5, 1e-12)
end)

test("torch.linalg.norm supports zero, negative, and infinite orders", function()
  local vector = torch.tensor({ 3, 4 }, { dtype = torch.float64 })
  assert_near(torch.linalg.norm(vector, 0):item(), 2, 1e-12)
  assert_near(torch.linalg.norm(vector, -1):item(), 12 / 7, 1e-12)
  assert_near(torch.linalg.norm(vector, -2):item(), 12 / 5, 1e-12)
  assert_near(torch.linalg.norm(vector, math.huge):item(), 4, 1e-12)
  assert_near(torch.linalg.norm(vector, -math.huge):item(), 3, 1e-12)

  local zeros = torch.tensor({ -2, 0, 1 }, { dtype = torch.float64 })
  assert_near(torch.linalg.norm(zeros, 0):item(), 2, 1e-12)
  assert_near(torch.linalg.norm(zeros, -math.huge):item(), 0, 1e-12)
  assert_near(torch.linalg.vector_norm(vector, -1):item(), 12 / 7, 1e-12)
  assert_shape(torch.linalg.vector_norm(vector, 0, nil, true), { 1 })
  assert_near(torch.linalg.vector_norm(vector, -math.huge):item(), 3, 1e-12)
end)

test("torch.linalg.norm validates dispatch and dimensions", function()
  local matrix = float_matrix({ { 1, 2 }, { 3, 4 } })
  assert_error(function()
    torch.linalg.norm(1)
  end, "expects a tensor")
  assert_error(function()
    torch.linalg.norm(torch.ones(2, 2, 2), 2)
  end, "1-D or 2-D")
  assert_error(function()
    torch.linalg.norm(matrix, nil, { 1, 2, 3 })
  end, "1- or 2-element table")
  assert_error(function()
    torch.linalg.norm(matrix, nil, "rows")
  end, "integer or a table")
  assert_error(function()
    torch.linalg.norm(matrix, nil, 0)
  end, "out of range")
  assert_error(function()
    torch.linalg.norm(matrix, nil, 3)
  end, "out of range")
  assert_error(function()
    torch.linalg.norm(matrix, "fro", 1)
  end, "ord must be a number")
  assert_error(function()
    torch.linalg.norm(matrix, 3, { 1, 2 })
  end, "matrix_norm supports")
  assert_error(function()
    torch.linalg.vector_norm(matrix, "fro")
  end, "ord must be a number")
end)

test("lu_factor_ex returns compact factors and info", function()
  local a = float_matrix({ { 4, 3 }, { 2, 1 } })
  local lu, pivots, info = a:lu_factor_ex()
  local reference_lu, reference_pivots = a:lu_factor()
  assert_tensor(lu, reference_lu:totable(), 0)
  assert_tensor(pivots, reference_pivots:totable(), 0)
  assert_dtype(lu, "float64")
  assert_dtype(pivots, "int32")
  assert_dtype(info, "int32")
  assert_shape(info, {})
  assert_eq(info:item(), 0)

  local alias_lu, alias_pivots, alias_info = torch.linalg.lu_factor_ex(a)
  assert_tensor(alias_lu, lu:totable(), 0)
  assert_tensor(alias_pivots, pivots:totable(), 0)
  assert_eq(alias_info:item(), 0)

  local tall = float_matrix({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local tall_lu, tall_pivots, tall_info = tall:lu_factor_ex()
  local tall_reference_lu, tall_reference_pivots = tall:lu_factor()
  assert_tensor(tall_lu, tall_reference_lu:totable(), 0)
  assert_tensor(tall_pivots, tall_reference_pivots:totable(), 0)
  assert_eq(tall_info:item(), 0)

  local batch = torch.tensor({
    { { 3, 1 }, { 1, 2 } },
    { { 4, 2 }, { 1, 3 } },
  }, { dtype = torch.float64 })
  local batch_lu, batch_pivots, batch_info = batch:lu_factor_ex()
  assert_shape(batch_lu, { 2, 2, 2 })
  assert_shape(batch_pivots, { 2, 2 })
  assert_shape(batch_info, { 2 })
  assert_tensor(batch_info, { 0, 0 }, 0)

  local empty_lu, empty_pivots, empty_info =
    torch.zeros(0, 0, { dtype = torch.float64 }):lu_factor_ex()
  assert_shape(empty_lu, { 0, 0 })
  assert_shape(empty_pivots, { 0 })
  assert_shape(empty_info, {})
  assert_eq(empty_info:item(), 0)

  -- Disabling pivoting has no CPU LAPACK kernel; reject it explicitly
  -- instead of leaking ATen's implementation-detail message.
  local swapped = float_matrix({ { 2, 1 }, { 4, 3 } })
  assert_error(function()
    swapped:lu_factor_ex({ pivot = false })
  end, "CPU backend")

  local gradient_a = torch.tensor({ { 4, 1 }, { 2, 3 } }, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local gradient_lu, gradient_pivots, gradient_info =
    gradient_a:lu_factor_ex()
  assert_true(gradient_lu:requires_grad(), "LU carries gradients")
  assert_true(not gradient_pivots:requires_grad(), "pivots stay detached")
  assert_true(not gradient_info:requires_grad(), "info stays detached")
end)

test("lu_factor_ex reports singular matrices and validates inputs", function()
  local singular = float_matrix({ { 1, 2 }, { 2, 4 } })
  local lu, pivots, info = singular:lu_factor_ex()
  assert_shape(lu, { 2, 2 })
  assert_shape(pivots, { 2 })
  assert_dtype(info, "int32")
  assert_eq(info:item(), 2)
  assert_error(function()
    singular:lu_factor_ex({ check_errors = true })
  end, "is zero")
  assert_error(function()
    singular:lu_factor()
  end, "is zero")

  assert_error(function()
    torch.ones(3):lu_factor_ex()
  end, "matrix or a batch")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):lu_factor_ex()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float16 }):lu_factor_ex()
  end, "float32 and float64")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):lu_factor_ex("options")
  end, "options must be a table")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):lu_factor_ex({ transpose = true })
  end, "unknown option")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):lu_factor_ex({ pivot = 1 })
  end, "pivot must be a boolean")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):lu_factor_ex({
      check_errors = "yes",
    })
  end, "check_errors must be a boolean")
end)

test("solve_ex solves dense systems and returns info", function()
  local a = float_matrix({ { 4, 3 }, { 2, 1 } })
  local b = float_matrix({ { 1 }, { 2 } })
  local solution, info = a:solve_ex(b)
  assert_tensor(solution, a:solve(b):totable(), 1e-12)
  assert_dtype(solution, "float64")
  assert_dtype(info, "int32")
  assert_shape(info, {})
  assert_eq(info:item(), 0)

  local alias_solution, alias_info = torch.linalg.solve_ex(a, b)
  assert_tensor(alias_solution, solution:totable(), 1e-12)
  assert_eq(alias_info:item(), 0)

  -- 1-D right-hand sides are supported for a left solve.
  local vector = torch.tensor({ 1, 2 }, { dtype = torch.float64 })
  local vector_solution, vector_info = a:solve_ex(vector)
  assert_shape(vector_solution, { 2 })
  assert_tensor(vector_solution, a:solve(vector):totable(), 1e-12)
  assert_eq(vector_info:item(), 0)

  -- left=false solves X A = B with a matrix right-hand side.
  local right = float_matrix({ { 1, 2 }, { 3, 4 } })
  local right_solution, right_info = a:solve_ex(right, { left = false })
  assert_tensor(right_solution, right:matmul(a:inverse()):totable(), 1e-12)
  assert_eq(right_info:item(), 0)

  local batch = torch.tensor({
    { { 4, 1 }, { 2, 3 } },
    { { 3, 1 }, { 1, 2 } },
  }, { dtype = torch.float64 })
  local rhs = torch.tensor({
    { { 1 }, { 2 } },
    { { 2 }, { 1 } },
  }, { dtype = torch.float64 })
  local batch_solution, batch_info = batch:solve_ex(rhs)
  assert_shape(batch_solution, { 2, 2, 1 })
  assert_shape(batch_info, { 2 })
  assert_tensor(batch_solution, batch:solve(rhs):totable(), 1e-12)
  assert_tensor(batch_info, { 0, 0 }, 0)

  -- A right-hand side without batch dimensions broadcasts over the batch.
  local shared, shared_info = batch:solve_ex(b)
  assert_tensor(shared, batch:solve(b):totable(), 1e-12)
  assert_tensor(shared_info, { 0, 0 }, 0)
end)

test("solve_ex gradients match the ordinary solve", function()
  local values = { { 4, 1 }, { 2, 3 } }
  local rhs_values = { { 1, 2 }, { 3, 4 } }
  local weight = float_matrix({ { 1, 2 }, { 3, 5 } })

  local a = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local b = torch.tensor(rhs_values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local solution, info = a:solve_ex(b)
  assert_true(solution:requires_grad(), "solution carries gradients")
  assert_true(not info:requires_grad(), "info stays detached")
  local loss = (solution * weight):sum()
  loss:backward()

  local reference_a = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local reference_b = torch.tensor(rhs_values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local reference_loss = (reference_a:solve(reference_b) * weight):sum()
  reference_loss:backward()
  assert_near(a:grad():totable(), reference_a:grad():totable(), 1e-10)
  assert_near(b:grad():totable(), reference_b:grad():totable(), 1e-10)
end)

test("solve_ex reports singular matrices and validates inputs", function()
  local singular = float_matrix({ { 1, 2 }, { 2, 4 } })
  local rhs = float_matrix({ { 1 }, { 2 } })
  local solution, info = singular:solve_ex(rhs)
  assert_shape(solution, { 2, 1 })
  assert_dtype(info, "int32")
  assert_eq(info:item(), 2)
  assert_error(function()
    singular:solve_ex(rhs, { check_errors = true })
  end, "singular")

  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):solve_ex({ 1, 2 })
  end, "tensor right-hand side")
  assert_error(function()
    torch.ones(2, 3, { dtype = torch.float64 }):solve_ex(rhs)
  end, "square matrices")
  assert_error(function()
    torch.ones(2, { dtype = torch.float64 }):solve_ex(rhs)
  end, "square matrices")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):solve_ex(rhs)
  end, "float32 and float64")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):solve_ex(
      torch.ones(2, 1, { dtype = torch.float32 }))
  end, "match the matrix dtype")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):solve_ex(
      torch.ones(3, { dtype = torch.float64 }))
  end, "wrong length")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):solve_ex(
      torch.ones(3, 1, { dtype = torch.float64 }))
  end, "wrong number of rows")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):solve_ex(
      torch.ones(2, { dtype = torch.float64 }), { left = false })
  end, "at least 2 dimensions")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):solve_ex(
      torch.ones(1, 3, { dtype = torch.float64 }), { left = false })
  end, "wrong number of columns")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):solve_ex(rhs, "options")
  end, "options must be a table")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):solve_ex(rhs, { transpose = true })
  end, "unknown option")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):solve_ex(rhs, { left = 1 })
  end, "left must be a boolean")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):solve_ex(rhs, {
      check_errors = "yes",
    })
  end, "check_errors must be a boolean")
end)

test("ldl_factor returns compact symmetric factors", function()
  local a = float_matrix({ { 4, 2 }, { 2, 3 } })
  local ld, pivots = a:ldl_factor()
  assert_tensor(ld, { { 4, 0 }, { 0.5, 2 } }, 1e-12)
  assert_tensor(pivots, { 1, 2 }, 0)
  assert_dtype(ld, "float64")
  assert_dtype(pivots, "int32")
  assert_shape(pivots, { 2 })

  -- Rebuild A = L D L^T for this 1x1-pivot factorization.
  local l = torch.zeros(2, 2, { dtype = torch.float64 })
  local d = torch.zeros(2, 2, { dtype = torch.float64 })
  for i = 1, 2 do
    l:set(i, i, 1)
    d:set(i, i, ld:get(i, i))
    for j = 1, i - 1 do l:set(i, j, ld:get(i, j)) end
  end
  assert_tensor(l:matmul(d):matmul(l:t()), a:totable(), 1e-12)

  local alias_ld, alias_pivots = torch.linalg.ldl_factor(a)
  assert_tensor(alias_ld, ld:totable(), 0)
  assert_tensor(alias_pivots, pivots:totable(), 0)

  -- For real inputs hermitian is accepted and has no effect.
  local hermitian_ld, hermitian_pivots = a:ldl_factor({ hermitian = true })
  assert_tensor(hermitian_ld, ld:totable(), 0)
  assert_tensor(hermitian_pivots, pivots:totable(), 0)

  local batch = torch.tensor({
    { { 4, 2 }, { 2, 3 } },
    { { 2, 1 }, { 1, 2 } },
  }, { dtype = torch.float64 })
  local batch_ld, batch_pivots = batch:ldl_factor()
  assert_shape(batch_ld, { 2, 2, 2 })
  assert_shape(batch_pivots, { 2, 2 })
  assert_tensor(batch_ld:select(1, 1), ld:totable(), 1e-12)
  assert_tensor(batch_ld:select(1, 2), { { 2, 0 }, { 0.5, 1.5 } }, 1e-12)
  assert_tensor(batch_pivots, { { 1, 2 }, { 1, 2 } }, 0)

  local empty_ld, empty_pivots =
    torch.zeros(0, 0, { dtype = torch.float64 }):ldl_factor()
  assert_shape(empty_ld, { 0, 0 })
  assert_shape(empty_pivots, { 0 })
  assert_dtype(empty_pivots, "int32")

  assert_dtype(torch.eye(2):ldl_factor(), "float32")
end)

test("ldl_factor validates shapes, dtypes, and options", function()
  assert_error(function()
    torch.ones(3):ldl_factor()
  end, "matrix or a batch")
  assert_error(function()
    torch.ones(2, 3, { dtype = torch.float64 }):ldl_factor()
  end, "square matrices")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):ldl_factor()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float16 }):ldl_factor()
  end, "float32 and float64")
  assert_error(function()
    torch.tensor({ { true, false }, { false, true } }):ldl_factor()
  end, "float32 and float64")
  assert_error(function()
    float_matrix({ { 4, 2 }, { 2, 3 } }):ldl_factor("options")
  end, "options must be a table")
  assert_error(function()
    float_matrix({ { 4, 2 }, { 2, 3 } }):ldl_factor({ upper = true })
  end, "unknown option")
  assert_error(function()
    float_matrix({ { 4, 2 }, { 2, 3 } }):ldl_factor({ hermitian = 1 })
  end, "hermitian must be a boolean")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):ldl_factor()
  end, "singular")
end)

test("ldl_factor_ex returns factors and info", function()
  local a = float_matrix({ { 4, 2 }, { 2, 3 } })
  local ld, pivots, info = a:ldl_factor_ex()
  local reference_ld, reference_pivots = a:ldl_factor()
  assert_tensor(ld, reference_ld:totable(), 0)
  assert_tensor(pivots, reference_pivots:totable(), 0)
  assert_dtype(ld, "float64")
  assert_dtype(pivots, "int32")
  assert_dtype(info, "int32")
  assert_shape(info, {})
  assert_eq(info:item(), 0)

  local alias_ld, alias_pivots, alias_info = torch.linalg.ldl_factor_ex(a)
  assert_tensor(alias_ld, ld:totable(), 0)
  assert_tensor(alias_pivots, pivots:totable(), 0)
  assert_eq(alias_info:item(), 0)

  local hermitian_ld, hermitian_pivots, hermitian_info =
    a:ldl_factor_ex({ hermitian = true })
  assert_tensor(hermitian_ld, ld:totable(), 0)
  assert_tensor(hermitian_pivots, pivots:totable(), 0)
  assert_eq(hermitian_info:item(), 0)

  local batch = torch.tensor({
    { { 4, 2 }, { 2, 3 } },
    { { 2, 1 }, { 1, 2 } },
  }, { dtype = torch.float64 })
  local batch_ld, batch_pivots, batch_info = batch:ldl_factor_ex()
  assert_shape(batch_ld, { 2, 2, 2 })
  assert_shape(batch_pivots, { 2, 2 })
  assert_shape(batch_info, { 2 })
  assert_tensor(batch_info, { 0, 0 }, 0)

  local empty_ld, empty_pivots, empty_info =
    torch.zeros(0, 0, { dtype = torch.float64 }):ldl_factor_ex()
  assert_shape(empty_ld, { 0, 0 })
  assert_shape(empty_pivots, { 0 })
  assert_shape(empty_info, {})
  assert_eq(empty_info:item(), 0)
end)

test("ldl_factor_ex reports singular matrices and validates inputs", function()
  local singular = float_matrix({ { 1, 2 }, { 2, 4 } })
  local ld, pivots, info = singular:ldl_factor_ex()
  assert_tensor(ld, { { 4, 0 }, { 0.5, 0 } }, 1e-12)
  assert_tensor(pivots, { 2, 2 }, 0)
  assert_dtype(info, "int32")
  assert_eq(info:item(), 2)
  assert_error(function()
    singular:ldl_factor_ex({ check_errors = true })
  end, "singular")

  assert_error(function()
    torch.ones(3):ldl_factor_ex()
  end, "matrix or a batch")
  assert_error(function()
    torch.ones(2, 3, { dtype = torch.float64 }):ldl_factor_ex()
  end, "square matrices")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):ldl_factor_ex()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float16 }):ldl_factor_ex()
  end, "float32 and float64")
  assert_error(function()
    float_matrix({ { 4, 2 }, { 2, 3 } }):ldl_factor_ex("options")
  end, "options must be a table")
  assert_error(function()
    float_matrix({ { 4, 2 }, { 2, 3 } }):ldl_factor_ex({ upper = true })
  end, "unknown option")
  assert_error(function()
    float_matrix({ { 4, 2 }, { 2, 3 } }):ldl_factor_ex({ hermitian = 1 })
  end, "hermitian must be a boolean")
  assert_error(function()
    float_matrix({ { 4, 2 }, { 2, 3 } }):ldl_factor_ex({
      check_errors = "yes",
    })
  end, "check_errors must be a boolean")
end)

test("ldl_solve solves systems from an LDL factorization", function()
  local a = float_matrix({
    { 4, 2, 0 },
    { 2, 3, 1 },
    { 0, 1, 2 },
  })
  local ld, pivots, info = a:ldl_factor_ex()
  assert_eq(info:item(), 0)
  local b = float_matrix({ { 1 }, { 2 }, { 3 } })
  local solution = ld:ldl_solve(pivots, b)
  assert_tensor(solution, { { 0.25 }, { 0 }, { 1.5 } }, 1e-12)
  assert_tensor(solution, a:solve(b):totable(), 1e-12)
  assert_dtype(solution, "float64")

  local alias_solution = torch.linalg.ldl_solve(ld, pivots, b)
  assert_tensor(alias_solution, solution:totable(), 1e-12)

  local multiple = float_matrix({
    { 1, 2 },
    { 2, 3 },
    { 3, 4 },
  })
  assert_tensor(ld:ldl_solve(pivots, multiple),
    a:solve(multiple):totable(), 1e-12)

  -- For real inputs the hermitian flag has no effect on the solve.
  assert_tensor(ld:ldl_solve(pivots, b, { hermitian = true }),
    solution:totable(), 1e-12)

  local batch = torch.tensor({
    { { 4, 2 }, { 2, 3 } },
    { { 2, 1 }, { 1, 2 } },
  }, { dtype = torch.float64 })
  local batch_ld, batch_pivots = batch:ldl_factor()
  local batch_rhs = torch.tensor({
    { { 1 }, { 2 } },
    { { 3 }, { 4 } },
  }, { dtype = torch.float64 })
  local batch_solution = batch_ld:ldl_solve(batch_pivots, batch_rhs)
  assert_shape(batch_solution, { 2, 2, 1 })
  assert_tensor(batch_solution, batch:solve(batch_rhs):totable(), 1e-12)

  -- A right-hand side without batch dimensions broadcasts over the factor.
  local shared, shared_info = batch:solve_ex(float_matrix({ { 1 }, { 2 } }))
  assert_tensor(shared_info, { 0, 0 }, 0)
  assert_tensor(batch_ld:ldl_solve(batch_pivots,
    float_matrix({ { 1 }, { 2 } })), shared:totable(), 1e-12)
end)

test("ldl_solve validates factors, pivots, and right-hand sides", function()
  local a = float_matrix({ { 4, 2 }, { 2, 3 } })
  local ld, pivots = a:ldl_factor()
  local rhs = float_matrix({ { 1 }, { 2 } })

  assert_error(function()
    ld:ldl_solve(2, rhs)
  end, "int32 pivots tensor")
  assert_error(function()
    ld:ldl_solve(pivots, { 1, 2 })
  end, "tensor right-hand side")
  assert_error(function()
    torch.ones(2, 3, { dtype = torch.float64 }):ldl_solve(pivots, rhs)
  end, "square matrices")
  assert_error(function()
    torch.ones(2, { dtype = torch.float64 }):ldl_solve(pivots, rhs)
  end, "square matrices")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):ldl_solve(pivots, rhs)
  end, "float32 and float64")
  assert_error(function()
    ld:ldl_solve(pivots:float(), rhs)
  end, "int32 dtype")
  assert_error(function()
    ld:ldl_solve(pivots, torch.ones(2, 1, { dtype = torch.float32 }))
  end, "match the matrix dtype")
  assert_error(function()
    ld:ldl_solve(torch.zeros(1, 2, { dtype = torch.int32 }), rhs)
  end, "one dimension less")
  assert_error(function()
    ld:ldl_solve(torch.zeros(1, { dtype = torch.int32 }), rhs)
  end, "wrong length")
  assert_error(function()
    ld:ldl_solve(pivots, float_matrix({ 1, 2 }))
  end, "at least 2 dimensions")
  assert_error(function()
    ld:ldl_solve(pivots, float_matrix({ { 1 }, { 2 }, { 3 } }))
  end, "wrong number of rows")
  assert_error(function()
    ld:ldl_solve(pivots, rhs, "options")
  end, "options must be a table")
  assert_error(function()
    ld:ldl_solve(pivots, rhs, { left = false })
  end, "unknown option")
  assert_error(function()
    ld:ldl_solve(pivots, rhs, { hermitian = 1 })
  end, "hermitian must be a boolean")

  local batch_ld, batch_pivots = torch.tensor({
    { { 4, 2 }, { 2, 3 } },
    { { 2, 1 }, { 1, 2 } },
  }, { dtype = torch.float64 }):ldl_factor()
  assert_error(function()
    batch_ld:ldl_solve(torch.zeros(3, 2, { dtype = torch.int32 }), rhs)
  end, "batch dimensions")
  assert_error(function()
    batch_ld:ldl_solve(batch_pivots, float_matrix({ { 1 }, { 2 }, { 3 } }))
  end, "wrong number of rows")
end)
