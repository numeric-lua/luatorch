local torch = require("torch")

local function float_matrix(rows)
  return torch.tensor(rows, { dtype = torch.float64 })
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
