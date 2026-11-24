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

test("lu_unpack reconstructs P, L, U from a packed factorization", function()
  local a = float_matrix({ { 4, 3 }, { 2, 1 } })
  local lu, pivots = a:lu_factor()
  local p, l, u = lu:lu_unpack(pivots)
  assert_shape(p, { 2, 2 })
  assert_shape(l, { 2, 2 })
  assert_shape(u, { 2, 2 })
  assert_tensor(p, { { 1, 0 }, { 0, 1 } })
  assert_tensor(l, { { 1, 0 }, { 0.5, 1 } }, 1e-12)
  assert_tensor(u, { { 4, 3 }, { 0, -0.5 } }, 1e-12)
  assert_tensor(p:matmul(l):matmul(u), a:totable(), 1e-12)
  assert_true(not p:requires_grad(), "P is detached")

  local module_p, module_l, module_u = torch.lu_unpack(lu, pivots)
  assert_tensor(module_p, p:totable(), 1e-12)
  assert_tensor(module_l, l:totable(), 1e-12)
  assert_tensor(module_u, u:totable(), 1e-12)

  local linalg_p, linalg_l, linalg_u = torch.linalg.lu_unpack(lu, pivots)
  assert_tensor(linalg_p, p:totable(), 1e-12)
  assert_tensor(linalg_l, l:totable(), 1e-12)
  assert_tensor(linalg_u, u:totable(), 1e-12)

  local swapped = float_matrix({ { 0, 1 }, { 1, 0 } })
  local swapped_lu, swapped_pivots = swapped:lu_factor()
  local sp, sl, su = torch.lu_unpack(swapped_lu, swapped_pivots)
  assert_eq(swapped_pivots:totable(), { 2, 2 })
  assert_tensor(sp, { { 0, 1 }, { 1, 0 } })
  assert_tensor(sl, { { 1, 0 }, { 0, 1 } })
  assert_tensor(su, { { 1, 0 }, { 0, 1 } })
  assert_tensor(sp:matmul(sl):matmul(su), swapped:totable(), 1e-12)

  local only_p, empty_l, empty_u = torch.lu_unpack(lu, pivots, {
    unpack_data = false,
  })
  assert_tensor(only_p, p:totable(), 1e-12)
  assert_shape(empty_l, { 0 })
  assert_shape(empty_u, { 0 })
  local empty_p, only_l, only_u = torch.lu_unpack(lu, pivots, {
    unpack_pivots = false,
  })
  assert_shape(empty_p, { 0 })
  assert_tensor(only_l, l:totable(), 1e-12)
  assert_tensor(only_u, u:totable(), 1e-12)
end)

test("lu_unpack handles rectangular, batched, and empty factors", function()
  local tall = float_matrix({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local tall_lu, tall_pivots = tall:lu_factor()
  local tp, tl, tu = torch.lu_unpack(tall_lu, tall_pivots)
  assert_shape(tp, { 3, 3 })
  assert_shape(tl, { 3, 2 })
  assert_shape(tu, { 2, 2 })
  assert_tensor(tp:matmul(tl):matmul(tu), tall:totable(), 1e-12)

  local wide = float_matrix({ { 1, 2, 3 }, { 4, 5, 6 } })
  local wide_lu, wide_pivots = wide:lu_factor()
  local wp, wl, wu = torch.lu_unpack(wide_lu, wide_pivots)
  assert_shape(wp, { 2, 2 })
  assert_shape(wl, { 2, 2 })
  assert_shape(wu, { 2, 3 })
  assert_tensor(wp:matmul(wl):matmul(wu), wide:totable(), 1e-12)

  local batch = torch.tensor({
    { { 4, 1, 2 }, { 3, 5, 1 }, { 1, 2, 6 } },
    { { 0, 2, 1 }, { 1, 0, 3 }, { 4, 1, 0 } },
  }, { dtype = torch.float64 })
  local batch_lu, batch_pivots = batch:lu_factor()
  local bp, bl, bu = torch.lu_unpack(batch_lu, batch_pivots)
  assert_shape(bp, { 2, 3, 3 })
  assert_shape(bl, { 2, 3, 3 })
  assert_shape(bu, { 2, 3, 3 })
  for i = 1, 2 do
    assert_tensor(bp:select(1, i):matmul(bl:select(1, i)):matmul(
      bu:select(1, i)), batch:select(1, i):totable(), 1e-12)
  end

  local empty_lu, empty_pivots =
    torch.zeros(0, 0, { dtype = torch.float64 }):lu_factor()
  local ep, el, eu = torch.lu_unpack(empty_lu, empty_pivots)
  assert_shape(ep, { 0, 0 })
  assert_shape(el, { 0, 0 })
  assert_shape(eu, { 0, 0 })
end)

test("lu_unpack gradients flow through L and U", function()
  local values = { { 4, 1 }, { 2, 3 } }
  local weight_l = float_matrix({ { 1, 2 }, { 3, 5 } })
  local weight_u = float_matrix({ { -1, 0.5 }, { 2, 4 } })
  local a = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local lu, pivots = a:lu_factor()
  local p, l, u = lu:lu_unpack(pivots)
  assert_true(not p:requires_grad(), "P is detached")
  assert_true(l:requires_grad())
  assert_true(u:requires_grad())
  local loss = (l * weight_l):sum() + (u * weight_u):sum()
  loss:backward()

  local eps = 1e-6
  local function perturbed(i, j, delta)
    local copy = deep_copy(values)
    copy[i][j] = copy[i][j] + delta
    local factor = torch.tensor(copy, { dtype = torch.float64 })
    local packed, packed_pivots = factor:lu_factor()
    local _, pl, pu = torch.lu_unpack(packed, packed_pivots)
    return ((pl * weight_l):sum() + (pu * weight_u):sum()):item()
  end
  for i = 1, 2 do
    for j = 1, 2 do
      local numeric = (perturbed(i, j, eps) - perturbed(i, j, -eps)) /
        (2 * eps)
      assert_near(a:grad():get(i, j), numeric, 1e-5,
        string.format("d lu_unpack loss / d A[%d][%d]", i, j))
    end
  end
end)

test("lu_unpack validates shapes, dtypes, pivots, and options", function()
  local a = float_matrix({ { 4, 3 }, { 2, 1 } })
  local lu, pivots = a:lu_factor()
  assert_error(function()
    lu:lu_unpack({ 1, 2 })
  end, "int32 pivots tensor")
  assert_error(function()
    lu:lu_unpack(pivots:to(torch.float64))
  end, "int32 dtype")
  assert_error(function()
    lu:lu_unpack(pivots:narrow(1, 1, 1))
  end, "wrong length")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):lu_unpack(pivots)
  end, "float32 and float64")
  assert_error(function()
    torch.ones(3):lu_unpack(pivots)
  end, "matrix or a batch")
  assert_error(function()
    lu:lu_unpack(pivots, { unknown = true })
  end, "unknown option")
  assert_error(function()
    lu:lu_unpack(pivots, { unpack_data = 1 })
  end, "must be a boolean")
  local bad = pivots:clone()
  bad:set(1, 99)
  assert_error(function()
    lu:lu_unpack(bad)
  end, "1-based pivot indices")
  assert_error(function()
    torch.linalg.lu_unpack({ 1, 2 })
  end, "packed LU tensor")

  local batch = torch.tensor({
    { { 1, 0 }, { 0, 1 } },
    { { 1, 0 }, { 0, 1 } },
  }, { dtype = torch.float64 })
  local batch_lu, batch_pivots = batch:lu_factor()
  assert_error(function()
    batch_lu:lu_unpack(batch_pivots:narrow(1, 1, 1))
  end, "batch dimensions")
end)

test("vander builds decreasing and increasing matrices", function()
  local x = torch.tensor({ 1, 2, 3 }, { dtype = torch.float64 })
  assert_tensor(torch.vander(x), {
    { 1, 1, 1 },
    { 4, 2, 1 },
    { 9, 3, 1 },
  }, 1e-12)
  assert_tensor(torch.vander(x, 2), {
    { 1, 1 },
    { 2, 1 },
    { 3, 1 },
  }, 1e-12)
  assert_tensor(torch.vander(x, 4, true), {
    { 1, 1, 1, 1 },
    { 1, 2, 4, 8 },
    { 1, 3, 9, 27 },
  }, 1e-12)
  assert_shape(torch.vander(x, 0), { 3, 0 })

  assert_tensor(torch.linalg.vander(x), {
    { 1, 1, 1 },
    { 1, 2, 4 },
    { 1, 3, 9 },
  }, 1e-12)
  assert_tensor(torch.linalg.vander(x, 2), {
    { 1, 1 },
    { 1, 2 },
    { 1, 3 },
  }, 1e-12)
  assert_shape(torch.linalg.vander(torch.zeros(0)), { 0, 0 })
  assert_dtype(torch.linalg.vander(torch.ones(2, 2)), "float32")

  local integers = torch.tensor({ 2, 3 }, { dtype = torch.int64 })
  assert_tensor(torch.vander(integers), {
    { 2, 1 },
    { 3, 1 },
  })
  assert_dtype(torch.vander(integers), "int64")
end)

test("vander handles batches and gradients", function()
  local x = torch.tensor({
    { 1, 2, 3 },
    { 4, 5, 6 },
  }, { dtype = torch.float64 })
  local matrix = torch.linalg.vander(x)
  assert_shape(matrix, { 2, 3, 3 })
  assert_tensor(matrix:select(1, 1), {
    { 1, 1, 1 },
    { 1, 2, 4 },
    { 1, 3, 9 },
  }, 1e-12)
  assert_tensor(matrix:select(1, 2), {
    { 1, 4, 16 },
    { 1, 5, 25 },
    { 1, 6, 36 },
  }, 1e-12)

  local values = { 1.2, -0.7, 2.1 }
  local weight = float_matrix({
    { 1, 2, 3 },
    { -1, 0.5, 4 },
    { 2, -3, 1 },
  })
  local parameters = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local output = torch.linalg.vander(parameters)
  local loss = (output * weight):sum()
  loss:backward()
  local eps = 1e-6
  local function numeric(index, delta)
    local copy = { values[1], values[2], values[3] }
    copy[index] = copy[index] + delta
    local tensor = torch.tensor(copy, { dtype = torch.float64 })
    return (torch.linalg.vander(tensor) * weight):sum():item()
  end
  for i = 1, 3 do
    assert_near(parameters:grad():get(i),
      (numeric(i, eps) - numeric(i, -eps)) / (2 * eps), 1e-5)
  end
end)

test("vander validates lengths and dtypes", function()
  assert_error(function()
    torch.vander(torch.ones(2, 2))
  end, "1-D")
  assert_error(function()
    torch.linalg.vander(torch.tensor(1.0))
  end, "at least one dimension")
  assert_error(function()
    torch.linalg.vander(torch.ones(3), -1)
  end, "at least 0")
  assert_error(function()
    torch.linalg.vander(torch.ones(3), 1.5)
  end, "integer")
  assert_error(function()
    torch.vander(torch.ones(3), 2, "yes")
  end, "boolean")
  assert_error(function()
    torch.linalg.vander(torch.tensor({ true, false }))
  end, "bool")
end)

test("cross computes 3-vector products along a dimension", function()
  local a = float_matrix({ { 1, 0, 0 }, { 0, 1, 0 } })
  local b = float_matrix({ { 0, 1, 0 }, { 1, 0, 0 } })
  assert_tensor(torch.cross(a, b), {
    { 0, 0, 1 },
    { 0, 0, -1 },
  }, 1e-12)
  assert_tensor(torch.linalg.cross(a, b), {
    { 0, 0, 1 },
    { 0, 0, -1 },
  }, 1e-12)

  local first = float_matrix({ { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } })
  local second = float_matrix({ { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 0 } })
  assert_tensor(torch.cross(first, second, 1), {
    { 0, 0, -1 },
    { -1, 0, 0 },
    { 0, -1, 0 },
  }, 1e-12)

  -- A leading batch dimension broadcasts against a single 3-vector.
  local batched = torch.tensor({
    { { 1, 0, 0 }, { 0, 1, 0 } },
    { { 0, 0, 1 }, { 1, 0, 0 } },
  }, { dtype = torch.float64 })
  local broadcast = torch.cross(batched,
    float_matrix({ { 0, 1, 0 } }):reshape(1, 1, 3))
  assert_shape(broadcast, { 2, 2, 3 })
  assert_tensor(broadcast:select(1, 1), {
    { 0, 0, 1 },
    { 0, 0, 0 },
  }, 1e-12)
  assert_tensor(broadcast:select(1, 2), {
    { -1, 0, 0 },
    { 0, 0, 1 },
  }, 1e-12)

  local integer_a = torch.tensor({ { 1, 0, 0 } }, { dtype = torch.int64 })
  local integer_b = torch.tensor({ { 0, 1, 0 } }, { dtype = torch.int64 })
  assert_tensor(torch.cross(integer_a, integer_b), { { 0, 0, 1 } })
  assert_dtype(torch.cross(integer_a, integer_b), "int64")
end)

test("cross supports autograd and validates its inputs", function()
  local values_a = { { 1.2, -0.4, 2.3 }, { 0.5, 1.1, -0.7 } }
  local values_b = { { -0.3, 0.8, 1.4 }, { 2.0, -1.2, 0.6 } }
  local a = torch.tensor(values_a, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local b = torch.tensor(values_b, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local result = torch.cross(a, b)
  assert_true(result:requires_grad())
  result:sum():backward()
  assert_shape(a:grad(), { 2, 3 })
  assert_shape(b:grad(), { 2, 3 })

  local eps = 1e-6
  local function numeric(side, i, j, delta)
    local copy = deep_copy(side == "a" and values_a or values_b)
    copy[i][j] = copy[i][j] + delta
    local ta = torch.tensor(side == "a" and copy or values_a,
      { dtype = torch.float64 })
    local tb = torch.tensor(side == "b" and copy or values_b,
      { dtype = torch.float64 })
    return torch.cross(ta, tb):sum():item()
  end
  for i = 1, 2 do
    for j = 1, 3 do
      assert_near(a:grad():get(i, j),
        (numeric("a", i, j, eps) - numeric("a", i, j, -eps)) / (2 * eps),
        1e-5)
      assert_near(b:grad():get(i, j),
        (numeric("b", i, j, eps) - numeric("b", i, j, -eps)) / (2 * eps),
        1e-5)
    end
  end

  assert_error(function()
    torch.cross(torch.ones(2, 3), torch.ones(2, 2))
  end, "length 3")
  assert_error(function()
    torch.cross(torch.ones(3), torch.ones(1, 3))
  end, "same number of dimensions")
  assert_error(function()
    torch.cross(torch.ones(2, 3), torch.ones(2, 3, { dtype = torch.int64 }))
  end, "same dtype")
  assert_error(function()
    torch.cross(torch.ones(2, 3, { dtype = torch.bool }),
      torch.ones(2, 3, { dtype = torch.bool }))
  end, "bool")
  assert_error(function()
    torch.cross(torch.ones(2, 3), torch.ones(3, 3))
  end, "not broadcastable")
  assert_error(function()
    torch.cross(torch.ones(2, 3), torch.ones(2, 3), 3)
  end, "dimension out of range")
end)

test("diagonal extracts main and offset diagonals", function()
  local matrix = float_matrix({
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 },
  })
  assert_tensor(matrix:diagonal(), { 1, 5, 9 })
  assert_tensor(matrix:diagonal(1), { 2, 6 })
  assert_tensor(matrix:diagonal(2), { 3 })
  assert_tensor(matrix:diagonal(-1), { 4, 8 })
  assert_tensor(matrix:diagonal(-2), { 7 })
  assert_shape(matrix:diagonal(3), { 0 })
  assert_tensor(torch.diagonal(matrix), { 1, 5, 9 })
  assert_tensor(torch.linalg.diagonal(matrix), { 1, 5, 9 })

  local rectangular = float_matrix({
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 },
  })
  assert_tensor(rectangular:diagonal(1), { 2, 7 })
  assert_tensor(rectangular:diagonal(-1), { 5 })
  assert_tensor(rectangular:diagonal(0, 2, 1), { 1, 6 })
  assert_tensor(rectangular:t():diagonal(), { 1, 6 })
end)

test("diagonal handles batches and arbitrary dimension pairs", function()
  local tensor = torch.tensor({
    { { 1, 2, 3 }, { 4, 5, 6 } },
    { { 7, 8, 9 }, { 10, 11, 12 } },
  }, { dtype = torch.float64 })
  assert_tensor(tensor:diagonal(), {
    { 1, 10 },
    { 2, 11 },
    { 3, 12 },
  })
  assert_tensor(tensor:diagonal(0, 1, 3), {
    { 1, 8 },
    { 4, 11 },
  })
  assert_shape(torch.diagonal(tensor), { 3, 2 })
  assert_shape(torch.linalg.diagonal(tensor), { 2, 2 })
  assert_tensor(torch.linalg.diagonal(tensor), {
    { 1, 5 },
    { 7, 11 },
  })
  assert_tensor(torch.linalg.diagonal(tensor, 1), {
    { 2, 6 },
    { 8, 12 },
  })

  local four = torch.tensor({
    { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } },
    { { { 9, 10 }, { 11, 12 } }, { { 13, 14 }, { 15, 16 } } },
  }, { dtype = torch.float64 })
  local diagonal = four:diagonal(0, 2, 4)
  assert_shape(diagonal, { 2, 2, 2 })
  assert_tensor(diagonal:select(1, 1), { { 1, 6 }, { 3, 8 } })
  assert_tensor(diagonal:select(1, 2), { { 9, 14 }, { 11, 16 } })

  local empty = torch.zeros(2, 0, 3, { dtype = torch.float64 })
  assert_shape(empty:diagonal(), { 3, 0 })
  assert_shape(torch.linalg.diagonal(empty), { 2, 0 })
end)

test("diagonal gradients flow to the selected entries", function()
  local matrix = torch.tensor({
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 },
  }, { dtype = torch.float64, requires_grad = true })
  local weights = torch.tensor({ 2, -1 }, { dtype = torch.float64 })
  local diagonal = matrix:diagonal(1)
  local loss = (diagonal * weights):sum()
  loss:backward()
  assert_tensor(matrix:grad(), {
    { 0, 2, 0 },
    { 0, 0, -1 },
    { 0, 0, 0 },
  })

  local batched = torch.tensor({
    { { 1, 2 }, { 3, 4 } },
    { { 5, 6 }, { 7, 8 } },
  }, { dtype = torch.float64, requires_grad = true })
  local batched_diagonal = batched:diagonal()
  batched_diagonal:sum():backward()
  assert_tensor(batched:grad(), {
    { { 1, 1 }, { 0, 0 } },
    { { 0, 0 }, { 1, 1 } },
  })
end)

test("diagonal validates dimensions, offsets, and dtypes", function()
  assert_error(function()
    torch.ones(3):diagonal()
  end, "at least two dimensions")
  assert_error(function()
    torch.ones(2, 2):diagonal(0, 1, 1)
  end, "must be different")
  assert_error(function()
    torch.ones(2, 2):diagonal(0.5)
  end, "integer")
  assert_error(function()
    torch.ones(2, 2):diagonal(0, 1, 3)
  end, "out of range")
  assert_error(function()
    torch.tensor({ { true, false }, { false, true } }):diagonal()
  end, "bool")
end)

test("multi_dot multiplies chains in an optimal order", function()
  local a = float_matrix({ { 1, 2 }, { 3, 4 } })
  local b = float_matrix({ { 5, 6 }, { 7, 8 } })
  local c = float_matrix({ { 9, 10 }, { 11, 12 } })
  assert_tensor(torch.linalg.multi_dot({ a, b }), a:matmul(b):totable(),
    1e-12)
  assert_tensor(torch.linalg.multi_dot({ a, b, c }),
    a:matmul(b):matmul(c):totable(), 1e-12)

  local first = torch.randn(3, 4, { dtype = torch.float64 })
  local middle = torch.randn(4, 5, { dtype = torch.float64 })
  local last = torch.randn(5, 2, { dtype = torch.float64 })
  assert_tensor(torch.linalg.multi_dot({ first, middle, last }),
    first:matmul(middle):matmul(last):totable(), 1e-12)

  local chain = {
    torch.randn(2, 3, { dtype = torch.float64 }),
    torch.randn(3, 4, { dtype = torch.float64 }),
    torch.randn(4, 5, { dtype = torch.float64 }),
    torch.randn(5, 2, { dtype = torch.float64 }),
  }
  local expected = chain[1]
  for i = 2, #chain do expected = expected:matmul(chain[i]) end
  assert_tensor(torch.linalg.multi_dot(chain), expected:totable(), 1e-12)
  assert_dtype(torch.linalg.multi_dot({
    torch.ones(2, 2), torch.ones(2, 2),
  }), "float32")
end)

test("multi_dot handles vector endpoints", function()
  local vector = float_matrix({ { 1, 2 } })[1]
  local matrix = float_matrix({ { 3, 4 }, { 5, 6 } })
  assert_shape(torch.linalg.multi_dot({ vector, matrix }), { 2 })
  assert_tensor(torch.linalg.multi_dot({ vector, matrix }), { 13, 16 },
    1e-12)
  assert_shape(torch.linalg.multi_dot({ matrix, vector }), { 2 })
  assert_tensor(torch.linalg.multi_dot({ matrix, vector }), { 11, 17 },
    1e-12)
  local scalar = torch.linalg.multi_dot({ vector, vector })
  assert_shape(scalar, {})
  assert_near(scalar:item(), 5, 1e-12)
  assert_near(torch.linalg.multi_dot({ vector, matrix, vector }):item(),
    45, 1e-12)
end)

test("multi_dot carries gradients and validates its inputs", function()
  local values = {
    { { 1, 2 }, { 3, 4 } },
    { { 5, 6 }, { 7, 8 } },
    { { 9, 10 }, { 11, 12 } },
  }
  local tensors = {}
  for i = 1, 3 do
    tensors[i] = torch.tensor(values[i], {
      dtype = torch.float64,
      requires_grad = true,
    })
  end
  local result = torch.linalg.multi_dot(tensors)
  result:sum():backward()
  for i = 1, 3 do
    assert_shape(tensors[i]:grad(), { 2, 2 })
  end

  local eps = 1e-6
  local function numeric(index, row, column, delta)
    local copies = deep_copy(values)
    copies[index][row][column] = copies[index][row][column] + delta
    local chain = {}
    for i = 1, 3 do
      chain[i] = torch.tensor(copies[i], { dtype = torch.float64 })
    end
    return torch.linalg.multi_dot(chain):sum():item()
  end
  for index = 1, 3 do
    for row = 1, 2 do
      for column = 1, 2 do
        local expected = (numeric(index, row, column, eps) -
          numeric(index, row, column, -eps)) / (2 * eps)
        assert_near(tensors[index]:grad():get(row, column), expected, 1e-5)
      end
    end
  end

  assert_error(function()
    torch.linalg.multi_dot({ tensors[1] })
  end, "at least two")
  assert_error(function()
    torch.linalg.multi_dot("nope")
  end, "table of tensors")
  assert_error(function()
    torch.linalg.multi_dot({ tensors[1], 3 })
  end, "entry 2")
  assert_error(function()
    torch.linalg.multi_dot({
      torch.ones(2, 2, 2), torch.ones(2, 2),
    })
  end, "first and last")
  assert_error(function()
    torch.linalg.multi_dot({
      torch.ones(2, 2), torch.ones(2), torch.ones(2, 2),
    })
  end, "middle tensors")
  assert_error(function()
    torch.linalg.multi_dot({ torch.ones(2, 2), torch.ones(3, 3) })
  end, "cannot be multiplied")
  assert_error(function()
    torch.linalg.multi_dot({
      torch.ones(2, 2, { dtype = torch.int64 }),
      torch.ones(2, 2, { dtype = torch.int64 }),
    })
  end, "float16, float32, and float64")
  assert_error(function()
    torch.linalg.multi_dot({
      torch.ones(2, 2, { dtype = torch.float64 }),
      torch.ones(2, 2, { dtype = torch.float32 }),
    })
  end, "share one dtype")
end)
