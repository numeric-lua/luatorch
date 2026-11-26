-- Kronecker products, stacking helpers, cartesian products, searchsorted,
-- and bucketize.

local torch = require("torch")

test("kron computes vector and matrix Kronecker products", function()
  local vector = torch.tensor({1, 2}, {dtype = torch.float64})
  assert_eq(vector:kron(torch.tensor({3, 4}, {dtype = torch.float64}))
    :totable(), {3, 4, 6, 8})

  local left = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
  local right = torch.tensor({{0, 5}, {6, 7}}, {dtype = torch.float64})
  local product = left:kron(right)
  assert_shape(product, {4, 4})
  assert_eq(product:totable(), {
    {0, 5, 0, 10},
    {6, 7, 12, 14},
    {0, 15, 0, 20},
    {18, 21, 24, 28},
  })
  assert_eq(torch.kron(left, right):totable(), product:totable())
end)

test("kron aligns ranks, handles scalars, and is differentiable", function()
  assert_eq(torch.tensor(2, {dtype = torch.float64}):kron(
    torch.tensor({1, 2}, {dtype = torch.float64})):totable(), {2, 4})
  local left = torch.tensor({1, 2}, {dtype = torch.float64})
  local right = torch.tensor({{1}, {2}}, {dtype = torch.float64})
  assert_shape(left:kron(right), {2, 2})

  local values = torch.tensor({{1, 2}, {3, 4}},
    {dtype = torch.float64, requires_grad = true})
  values:kron(torch.eye(2, {dtype = torch.float64})):sum():backward()
  assert_eq(values:grad():totable(), {{2, 2}, {2, 2}})

  assert_error(function() left:kron(1) end, "another tensor")
  assert_error(function() torch.kron(1, left) end, "expects a tensor")
end)

test("hstack and vstack concatenate and promote ranks", function()
  local a = torch.tensor({1, 2}, {dtype = torch.float64})
  local b = torch.tensor({3, 4}, {dtype = torch.float64})
  assert_eq(torch.hstack({a, b}):totable(), {1, 2, 3, 4})
  assert_eq(torch.vstack({a, b}):totable(), {{1, 2}, {3, 4}})
  assert_eq(torch.row_stack({a, b}):totable(), {{1, 2}, {3, 4}})

  local matrix = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
  local column = torch.tensor({{5}, {6}}, {dtype = torch.float64})
  assert_eq(torch.hstack({matrix, column}):totable(),
    {{1, 2, 5}, {3, 4, 6}})
  assert_eq(torch.vstack({
    torch.tensor(5, {dtype = torch.float64}),
    torch.tensor(7, {dtype = torch.float64}),
  }):totable(), {{5}, {7}})
end)

test("dstack and column_stack build depth and column layouts", function()
  local a = torch.tensor({1, 2}, {dtype = torch.float64})
  local b = torch.tensor({3, 4}, {dtype = torch.float64})
  assert_shape(torch.dstack({a, b}), {1, 2, 2})
  assert_eq(torch.dstack({a, b}):totable(), {{{1, 3}, {2, 4}}})
  assert_shape(torch.column_stack({a, b}), {2, 2})
  assert_eq(torch.column_stack({a, b}):totable(), {{1, 3}, {2, 4}})

  local matrix = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
  local other = torch.tensor({{5}, {6}}, {dtype = torch.float64})
  assert_eq(torch.column_stack({matrix, other}):totable(),
    {{1, 2, 5}, {3, 4, 6}})
end)

test("stacking helpers are differentiable and validate inputs", function()
  local a = torch.tensor({1, 2}, {dtype = torch.float64,
    requires_grad = true})
  local b = torch.tensor({3, 4}, {dtype = torch.float64,
    requires_grad = true})
  torch.vstack({a, b}):sum():backward()
  assert_eq(a:grad():totable(), {1, 1})
  assert_eq(b:grad():totable(), {1, 1})

  assert_error(function() torch.hstack({}) end, "non-empty")
  assert_error(function() torch.vstack({a, 1}) end, "entry 2")
  assert_error(function() torch.dstack("x") end, "non-empty")
end)

test("cartesian_prod enumerates combinations with the first input slowest", function()
  local product = torch.cartesian_prod(
    torch.tensor({1, 2}, {dtype = torch.float64}),
    torch.tensor({3, 4, 5}, {dtype = torch.float64}))
  assert_shape(product, {6, 2})
  assert_eq(product:totable(), {
    {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 5},
  })
  assert_shape(torch.cartesian_prod(torch.tensor({1, 2, 3})), {3, 1})
  assert_shape(torch.cartesian_prod(
    torch.tensor({}, {dtype = torch.float64}),
    torch.tensor({1, 2}, {dtype = torch.float64})), {0, 2})
end)

test("cartesian_prod is differentiable and validates inputs", function()
  local first = torch.tensor({1, 2}, {dtype = torch.float64,
    requires_grad = true})
  local product = torch.cartesian_prod(first,
    torch.tensor({3, 4}, {dtype = torch.float64}))
  product:sum():backward()
  assert_eq(first:grad():totable(), {2, 2})

  assert_error(function() torch.cartesian_prod() end, "at least one")
  assert_error(function() torch.cartesian_prod(torch.tensor(1)) end, "1-D")
  assert_error(function() torch.cartesian_prod(1) end, "1-D")
end)

test("searchsorted returns 1-based insertion positions", function()
  local sorted = torch.tensor({1, 3, 5}, {dtype = torch.float64})
  local values = torch.tensor({0, 1, 2, 3, 6}, {dtype = torch.float64})
  local left = torch.searchsorted(sorted, values)
  assert_eq(left:dtype(), "int64")
  assert_eq(left:totable(), {1, 1, 2, 2, 4})
  assert_eq(torch.searchsorted(sorted, values, true):totable(),
    {1, 2, 2, 3, 4})
  assert_eq(torch.searchsorted(sorted,
    torch.tensor(3, {dtype = torch.float64})):item(), 2)

  local empty = torch.tensor({}, {dtype = torch.float64})
  assert_eq(torch.searchsorted(empty, values):totable(), {1, 1, 1, 1, 1})
end)

test("searchsorted and bucketize validate their inputs", function()
  local sorted = torch.tensor({1, 3, 5})
  local values = torch.tensor({2, 4})
  assert_error(function()
    torch.searchsorted(torch.tensor({{1, 2}}), values)
  end, "1-D")
  assert_error(function() torch.searchsorted(sorted, 1) end, "two tensors")
  assert_error(function() torch.searchsorted(sorted, values, 1) end,
    "boolean")
  assert_error(function() torch.bucketize(values, torch.tensor({{1, 3}}))
  end, "1-D")
end)

test("bucketize mirrors searchsorted for any input shape", function()
  local boundaries = torch.tensor({1, 3, 5}, {dtype = torch.float64})
  local input = torch.tensor({{0, 1}, {3, 6}}, {dtype = torch.float64})
  assert_shape(torch.bucketize(input, boundaries), {2, 2})
  assert_eq(torch.bucketize(input, boundaries):totable(),
    {{1, 1}, {2, 4}})
  assert_eq(torch.bucketize(input, boundaries, true):totable(),
    {{1, 2}, {3, 4}})
end)
