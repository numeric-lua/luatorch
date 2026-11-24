-- Triangular helpers, trace, torch.diag, and the left=false solve variant.

local torch = require("torch")

test("tril zeroes above the diagonal and honors offsets", function()
  local a = torch.tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
    {dtype = torch.float64})
  assert_eq(a:tril():totable(), {{1, 0, 0}, {4, 5, 0}, {7, 8, 9}})
  assert_eq(a:tril(1):totable(), {{1, 2, 0}, {4, 5, 6}, {7, 8, 9}})
  assert_eq(a:tril(-1):totable(), {{0, 0, 0}, {4, 0, 0}, {7, 8, 0}})
  assert_eq(torch.tril(a):totable(), a:tril():totable())
  assert_eq(a:tril():dtype(), "float64")
end)

test("triu zeroes below the diagonal and honors offsets", function()
  local a = torch.tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
    {dtype = torch.float64})
  assert_eq(a:triu():totable(), {{1, 2, 3}, {0, 5, 6}, {0, 0, 9}})
  assert_eq(a:triu(1):totable(), {{0, 2, 3}, {0, 0, 6}, {0, 0, 0}})
  assert_eq(a:triu(-1):totable(), {{1, 2, 3}, {4, 5, 6}, {0, 8, 9}})
  assert_eq(torch.triu(a):totable(), a:triu():totable())
end)

test("tril and triu support batches, integer/bool dtypes, and empty shapes", function()
  local batch = torch.tensor({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}},
    {dtype = torch.float64})
  assert_eq(batch:tril():totable(), {{{1, 0}, {3, 4}}, {{5, 0}, {7, 8}}})
  assert_eq(batch:triu():totable(), {{{1, 2}, {0, 4}}, {{5, 6}, {0, 8}}})

  local integers = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.int64})
  assert_eq(integers:tril():totable(), {{1, 0}, {3, 4}})
  assert_eq(integers:tril():dtype(), "int64")

  local flags = torch.tensor({{true, false}, {true, true}})
  assert_eq(flags:tril():totable(), {{true, false}, {true, true}})
  assert_eq(flags:triu():totable(), {{true, false}, {false, true}})
  assert_eq(flags:tril():dtype(), "bool")

  local empty = torch.zeros(0, 3, {dtype = torch.float64})
  assert_shape(empty:tril(), {0, 3})
  assert_shape(empty:triu(), {0, 3})
end)

test("tril and triu are differentiable with zero gradient in the zeroed half", function()
  local lower = torch.tensor({{1, 2}, {3, 4}},
    {dtype = torch.float64, requires_grad = true})
  lower:tril():sum():backward()
  assert_true(lower:grad():allclose(torch.tensor({{1, 0}, {1, 1}},
    {dtype = torch.float64})))

  local upper = torch.tensor({{1, 2}, {3, 4}},
    {dtype = torch.float64, requires_grad = true})
  upper:triu():sum():backward()
  assert_true(upper:grad():allclose(torch.tensor({{1, 1}, {0, 1}},
    {dtype = torch.float64})))
end)

test("tril and triu validate their inputs", function()
  local vector = torch.tensor({1, 2, 3})
  assert_error(function() vector:tril() end, "at least two dimensions")
  assert_error(function() vector:triu() end, "at least two dimensions")
  local matrix = torch.tensor({{1, 2}, {3, 4}})
  assert_error(function() matrix:tril(1.5) end, "diagonal")
  assert_error(function() matrix:triu("x") end, "diagonal")
  assert_error(function() torch.tril(1) end, "expects a tensor")
  assert_error(function() torch.triu({}) end, "expects a tensor")
end)

test("trace sums the main diagonal", function()
  local a = torch.tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
    {dtype = torch.float64})
  local total = a:trace()
  assert_eq(total:item(), 15)
  assert_eq(total:sizes(), {})
  assert_eq(torch.trace(a):item(), 15)

  local integers = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.int64})
  assert_eq(integers:trace():item(), 5)

  local empty = torch.zeros(0, 0, {dtype = torch.float64})
  assert_eq(empty:trace():item(), 0)

  assert_error(function() torch.tensor({{1, 2}, {3, 4}}):reshape(4):trace()
  end, "2-D")
  assert_error(function() torch.trace(1) end, "expects a tensor")
end)

test("trace is differentiable and passes gradients along the diagonal", function()
  local a = torch.tensor({{1, 2}, {3, 4}},
    {dtype = torch.float64, requires_grad = true})
  a:trace():backward()
  assert_true(a:grad():allclose(torch.tensor({{1, 0}, {0, 1}},
    {dtype = torch.float64})))
end)

test("diag extracts diagonals from matrices", function()
  local a = torch.tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
    {dtype = torch.float64})
  assert_eq(a:diag():totable(), {1, 5, 9})
  assert_eq(a:diag(1):totable(), {2, 6})
  assert_eq(a:diag(-1):totable(), {4, 8})
  assert_eq(a:diag(10):totable(), {})
  assert_eq(torch.diag(a):totable(), a:diag():totable())
  assert_true(a:diag(1):allclose(a:diagonal(1, 1, 2)))
end)

test("diag builds matrices from vectors", function()
  local v = torch.tensor({1, 2, 3}, {dtype = torch.float64})
  assert_eq(v:diag():totable(),
    {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}})
  assert_eq(v:diag(1):totable(),
    {{0, 1, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 3}, {0, 0, 0, 0}})
  assert_eq(v:diag(-1):totable(),
    {{0, 0, 0, 0}, {1, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 3, 0}})
  assert_eq(v:diag(-2):totable(),
    {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {1, 0, 0, 0, 0},
     {0, 2, 0, 0, 0}, {0, 0, 3, 0, 0}})
  assert_eq(torch.diag(v):totable(), v:diag():totable())

  local empty = torch.tensor({}, {dtype = torch.float64})
  assert_shape(empty:diag(), {0, 0})
  assert_shape(empty:diag(2), {2, 2})
  assert_eq(empty:diag(2):sum():item(), 0)
end)

test("diag supports bool/int data and is differentiable in both directions", function()
  local flags = torch.tensor({true, false, true})
  assert_eq(flags:diag():totable(),
    {{true, false, false}, {false, false, false}, {false, false, true}})
  assert_eq(flags:diag():dtype(), "bool")
  local integers = torch.tensor({1, 2}, {dtype = torch.int64})
  assert_eq(integers:diag():totable(), {{1, 0}, {0, 2}})

  local vector = torch.tensor({1, 2, 3},
    {dtype = torch.float64, requires_grad = true})
  vector:diag(1):sum():backward()
  assert_true(vector:grad():allclose(torch.ones(3,
    {dtype = torch.float64})))

  local matrix = torch.tensor({{1, 2}, {3, 4}},
    {dtype = torch.float64, requires_grad = true})
  matrix:diag():sum():backward()
  assert_true(matrix:grad():allclose(torch.tensor({{1, 0}, {0, 1}},
    {dtype = torch.float64})))
end)

test("diag validates its rank, diagonal, and alias argument", function()
  local cube = torch.zeros(2, 2, 2)
  assert_error(function() cube:diag() end, "1-D or 2-D")
  local matrix = torch.tensor({{1, 2}, {3, 4}})
  assert_error(function() matrix:diag(1.5) end, "diagonal")
  assert_error(function() matrix:diag(math.huge) end, "diagonal")
  assert_error(function() torch.diag(1) end, "expects a tensor")
end)

test("solve with left=false solves X A = B for batches and right sides", function()
  local a = torch.tensor({{2, 0}, {1, 3}}, {dtype = torch.float64})
  local b = torch.tensor({{2, 4}}, {dtype = torch.float64})
  local x = a:solve(b, { left = false })
  assert_true(x:allclose(torch.tensor({{1 / 3, 4 / 3}},
    {dtype = torch.float64}), 1e-12, 1e-12))
  assert_true(x:matmul(a):allclose(b, 1e-12, 1e-12))
  assert_true(torch.linalg.solve(a, b, { left = false }):allclose(x, 1e-12, 1e-12))

  local batch_a = torch.tensor({{{2, 0}, {1, 3}}, {{4, 0}, {0, 2}}},
    {dtype = torch.float64})
  local batch_b = torch.tensor({{{2, 4}}, {{8, 8}}},
    {dtype = torch.float64})
  local batch_x = batch_a:solve(batch_b, { left = false })
  assert_shape(batch_x, {2, 1, 2})
  assert_true(batch_x:matmul(batch_a):allclose(batch_b, 1e-12, 1e-12))

  -- The default remains a left solve.
  assert_true(a:solve(torch.tensor({{2}, {4}},
    {dtype = torch.float64})):allclose(
    torch.tensor({{1}, {1}}, {dtype = torch.float64}), 1e-12, 1e-12))
end)

test("solve with left=false is differentiable and validates its options", function()
  local a = torch.tensor({{2, 0}, {1, 3}},
    {dtype = torch.float64, requires_grad = true})
  local b = torch.tensor({{2, 4}},
    {dtype = torch.float64, requires_grad = true})
  local x = a:solve(b, { left = false })
  x:sum():backward()
  assert_true(b:grad():allclose(torch.tensor({{0.5, 1 / 6}},
    {dtype = torch.float64}), 1e-12, 1e-12))
  assert_true(a:grad():allclose(
    torch.tensor({{-1 / 6, -1 / 18}, {-2 / 3, -2 / 9}},
      {dtype = torch.float64}), 1e-12, 1e-12))

  assert_error(function() a:solve(b, { left = 1 }) end, "boolean")
  assert_error(function() a:solve(b, { foo = true }) end, "unknown option")
  assert_error(function() a:solve(b, "left") end, "options must be a table")
  assert_error(function()
    a:solve(torch.tensor({1, 2}, {dtype = torch.float64}), { left = false })
  end, "at least two dimensions")
  assert_error(function() a:solve(1) end, "right-hand side")
end)
