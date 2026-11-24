-- Matrix construction/contraction helpers: diag_embed, diagflat,
-- tril_indices/triu_indices, tensordot, and matrix_rank's rtol option.

local torch = require("torch")

test("diag_embed places a vector on the requested diagonal", function()
  local v = torch.tensor({1, 2, 3}, {dtype = torch.float64})
  assert_eq(v:diag_embed():totable(),
    {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}})
  assert_eq(v:diag_embed(1):totable(),
    {{0, 1, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 3}, {0, 0, 0, 0}})
  assert_eq(v:diag_embed(-1):totable(),
    {{0, 0, 0, 0}, {1, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 3, 0}})
  assert_eq(torch.diag_embed(v, 1):totable(), v:diag_embed(1):totable())
  assert_eq(v:diag_embed():dtype(), "float64")
end)

test("diag_embed treats leading dimensions as batches", function()
  local batch = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
  assert_eq(batch:diag_embed():totable(),
    {{{1, 0}, {0, 2}}, {{3, 0}, {0, 4}}})
  assert_eq(batch:diag_embed(-1):totable(),
    {{{0, 0, 0}, {1, 0, 0}, {0, 2, 0}},
     {{0, 0, 0}, {3, 0, 0}, {0, 4, 0}}})
  assert_shape(batch:diag_embed(1), {2, 3, 3})
end)

test("diag_embed supports bool/empty inputs and validates offsets", function()
  local flags = torch.tensor({true, false})
  assert_eq(flags:diag_embed():totable(),
    {{true, false}, {false, false}})
  assert_eq(flags:diag_embed():dtype(), "bool")

  local empty = torch.tensor({}, {dtype = torch.float64})
  assert_shape(empty:diag_embed(), {0, 0})
  assert_shape(empty:diag_embed(2), {2, 2})
  assert_eq(empty:diag_embed(2):sum():item(), 0)

  assert_error(function() torch.tensor(1):diag_embed() end, "at least one")
  assert_error(function() torch.tensor({1, 2}):diag_embed(1.5) end, "diagonal")
  assert_error(function() torch.diag_embed(1) end, "expects a tensor")
end)

test("diag_embed is differentiable with respect to the vector", function()
  local v = torch.tensor({1, 2, 3},
    {dtype = torch.float64, requires_grad = true})
  v:diag_embed(1):sum():backward()
  assert_true(v:grad():allclose(torch.ones(3,
    {dtype = torch.float64})))
end)

test("diagflat flattens its input and builds a diagonal matrix", function()
  local matrix = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
  assert_eq(torch.diagflat(matrix):totable(),
    {{1, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 3, 0}, {0, 0, 0, 4}})
  assert_eq(torch.diagflat(matrix, 1):totable(),
    {{0, 1, 0, 0, 0}, {0, 0, 2, 0, 0}, {0, 0, 0, 3, 0},
     {0, 0, 0, 0, 4}, {0, 0, 0, 0, 0}})
  assert_shape(torch.diagflat(torch.tensor({})), {0, 0})
  assert_error(function() torch.diagflat(1) end, "expects a tensor")
  assert_error(function() torch.diagflat(matrix, 1.5) end, "diagonal")
end)

test("diagflat is differentiable for every input entry", function()
  local matrix = torch.tensor({{1, 2}, {3, 4}},
    {dtype = torch.float64, requires_grad = true})
  torch.diagflat(matrix):sum():backward()
  assert_true(matrix:grad():allclose(torch.ones(2, 2,
    {dtype = torch.float64})))
end)

test("tril_indices returns 1-based coordinates in row-major order", function()
  local indices = torch.tril_indices(3, 3)
  assert_eq(indices:sizes(), {2, 6})
  assert_eq(indices:dtype(), "int64")
  assert_eq(indices:totable(),
    {{1, 2, 2, 3, 3, 3}, {1, 1, 2, 1, 2, 3}})
  assert_eq(torch.tril_indices(3, 3, -1):totable(),
    {{2, 3, 3}, {1, 1, 2}})
  assert_shape(torch.tril_indices(0, 5), {2, 0})
end)

test("triu_indices returns 1-based coordinates in row-major order", function()
  local indices = torch.triu_indices(3, 3)
  assert_eq(indices:sizes(), {2, 6})
  assert_eq(indices:totable(),
    {{1, 1, 1, 2, 2, 3}, {1, 2, 3, 2, 3, 3}})
  assert_eq(torch.triu_indices(3, 3, 1):totable(),
    {{1, 1, 2}, {2, 3, 3}})
  assert_eq(torch.triu_indices(3, 3, -1):totable(),
    {{1, 1, 1, 2, 2, 2, 3, 3}, {1, 2, 3, 1, 2, 3, 2, 3}})
  assert_shape(torch.triu_indices(4, 0), {2, 0})
end)

test("triangular index helpers support dtypes and validate arguments", function()
  local indices = torch.tril_indices(2, 2, nil, { dtype = torch.int32 })
  assert_eq(indices:dtype(), "int32")
  assert_eq(indices:totable(), {{1, 2, 2}, {1, 1, 2}})
  assert_error(function() torch.tril_indices(-1, 2) end, "row")
  assert_error(function() torch.triu_indices(2, 2.5) end, "col")
  assert_error(function() torch.tril_indices(2, 2, 0.5) end, "diagonal")
  assert_error(function() torch.tril_indices(2, 2, nil, { device = "cuda" })
  end, "CPU")
  assert_error(function() torch.tril_indices(2, 2, nil, { foo = true })
  end, "unknown option")
  assert_error(function() torch.triu_indices(2, 2, nil, 1) end, "table")
end)

test("tensordot contracts vectors, matrices, and outer products", function()
  local a = torch.tensor({1, 2}, {dtype = torch.float64})
  local b = torch.tensor({3, 4}, {dtype = torch.float64})
  assert_eq(torch.tensordot(a, b, 1):item(), 11)
  assert_eq(torch.tensordot(a, b, 0):totable(), {{3, 4}, {6, 8}})

  local m1 = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
  local m2 = torch.tensor({{5, 6}, {7, 8}}, {dtype = torch.float64})
  assert_true(torch.tensordot(m1, m2, 1):allclose(m1:matmul(m2)))
  assert_eq(torch.tensordot(m1, m2, 2):item(), 70)
end)

test("tensordot honors explicit and negative dimension lists", function()
  local x = torch.arange(24, {dtype = torch.float64}):reshape(2, 3, 4)
  local y = torch.arange(20, {dtype = torch.float64}):reshape(4, 5)
  local reference = torch.einsum("ijk,kl->ijl", x, y)

  local result = torch.tensordot(x, y, { { 3 }, { 1 } })
  assert_shape(result, {2, 3, 5})
  assert_true(result:allclose(reference))
  assert_true(torch.tensordot(x, y, { { -1 }, { -2 } }):allclose(reference))
  assert_true(torch.tensordot(x, y, 1):allclose(reference))
  assert_true(torch.tensordot(x, y, 0):allclose(
    torch.einsum("ijk,lm->ijklm", x, y)))
end)

test("tensordot is differentiable with respect to both operands", function()
  local a = torch.tensor({{1, 2}, {3, 4}},
    {dtype = torch.float64, requires_grad = true})
  local b = torch.tensor({{5, 6}, {7, 8}},
    {dtype = torch.float64, requires_grad = true})
  torch.tensordot(a, b, 1):sum():backward()
  assert_true(a:grad():allclose(torch.tensor({{11, 15}, {11, 15}},
    {dtype = torch.float64})))
  assert_true(b:grad():allclose(torch.tensor({{4, 4}, {6, 6}},
    {dtype = torch.float64})))
end)

test("tensordot validates dimensions, shapes, and dtypes", function()
  local a = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
  local b = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
  assert_error(function() torch.tensordot(1, b) end, "two tensors")
  assert_error(function() torch.tensordot(torch.tensor(1), b) end,
    "at least one")
  assert_error(function() torch.tensordot(a, b, 3) end, "exceeds")
  assert_error(function() torch.tensordot(a, b, -1) end, "non-negative")
  assert_error(function() torch.tensordot(a, b, { { 1, 1 }, { 1, 2 } })
  end, "repeat")
  assert_error(function() torch.tensordot(a, b, { { 1 }, { 1, 2 } })
  end, "same length")
  local wide = torch.tensor({{1, 2, 3}, {4, 5, 6}},
    {dtype = torch.float64})
  assert_error(function() torch.tensordot(a, wide, { { 1 }, { 2 } })
  end, "mismatched sizes")
  assert_error(function()
    torch.tensordot(a, torch.tensor({{1, 2}}, {dtype = torch.float32}), 1)
  end, "dtype")
end)

test("matrix_rank accepts rtol and tol options", function()
  local matrix = torch.diag(torch.tensor({4, 1, 0.1},
    {dtype = torch.float64}))
  assert_eq(torch.linalg.matrix_rank(matrix), 3)
  assert_eq(torch.linalg.matrix_rank(matrix, { rtol = 1e-6 }), 3)
  assert_eq(torch.linalg.matrix_rank(matrix, { rtol = 0.2 }), 2)
  assert_eq(torch.linalg.matrix_rank(matrix, { rtol = 0.5 }), 1)
  assert_eq(torch.linalg.matrix_rank(matrix, { rtol = 1 }), 0)
  assert_eq(torch.linalg.matrix_rank(matrix, { tol = 0.5 }), 2)
  -- The historical numeric tolerance and hermitian arguments still work.
  assert_eq(torch.linalg.matrix_rank(matrix, 0.5), 2)
  assert_eq(torch.linalg.matrix_rank(matrix, nil, true), 3)
  assert_eq(torch.linalg.matrix_rank(matrix,
    { hermitian = true, rtol = 0.2 }), 2)
end)

test("matrix_rank options apply to batches and validate their keys", function()
  local batch = torch.tensor({{{1, 0}, {0, 1}}, {{1, 0}, {0, 0}}},
    {dtype = torch.float64})
  assert_eq(torch.linalg.matrix_rank(batch, { rtol = 1e-6 }):totable(),
    {2, 1})
  assert_eq(torch.linalg.matrix_rank(batch,
    { hermitian = true, rtol = 1e-6 }):totable(), {2, 1})

  local matrix = torch.eye(2, {dtype = torch.float64})
  assert_error(function()
    torch.linalg.matrix_rank(matrix, { tol = 0.1, rtol = 0.1 })
  end, "either tol or rtol")
  assert_error(function()
    torch.linalg.matrix_rank(matrix, { foo = 1 })
  end, "unknown option")
  assert_error(function()
    torch.linalg.matrix_rank(matrix, { hermitian = 1 })
  end, "boolean")
  assert_error(function()
    torch.linalg.matrix_rank(matrix, {}, true)
  end, "cannot be combined")
end)
