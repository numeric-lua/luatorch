-- Splitting and axis utilities: split, chunk, unbind, swapaxes/moveaxis,
-- and atleast_1d/2d/3d.

local torch = require("torch")

test("split divides a dimension into fixed-size views", function()
  local matrix = torch.tensor({{1, 2, 3}, {4, 5, 6}},
    {dtype = torch.float64})
  local parts = matrix:split(2, 2)
  assert_eq(#parts, 2)
  assert_shape(parts[1], {2, 2})
  assert_shape(parts[2], {2, 1})
  assert_eq(parts[1]:totable(), {{1, 2}, {4, 5}})
  assert_eq(parts[2]:totable(), {{3}, {6}})
  assert_eq(#matrix:split(4, 2), 1)
  assert_eq(torch.split(matrix, 1, 1)[2]:totable(), {{4, 5, 6}})
end)

test("split accepts explicit section sizes and validates them", function()
  local matrix = torch.tensor({{1, 2, 3}, {4, 5, 6}},
    {dtype = torch.float64})
  local parts = matrix:split({1, 2}, 2)
  assert_eq(#parts, 2)
  assert_eq(parts[1]:totable(), {{1}, {4}})
  assert_eq(parts[2]:totable(), {{2, 3}, {5, 6}})
  assert_eq(#matrix:split({0, 3, 0}, 2), 3)
  assert_shape(matrix:split({0, 3, 0}, 2)[1], {2, 0})

  assert_error(function() matrix:split({1, 1}, 2) end, "sum")
  assert_error(function() matrix:split({-1, 4}, 2) end, "non-negative")
  assert_error(function() matrix:split(0, 2) end, "positive")
  assert_error(function() matrix:split(1.5, 2) end, "positive")
  assert_error(function() torch.split(1, 2) end, "expects a tensor")
end)

test("chunk splits into at most the requested number of parts", function()
  local matrix = torch.tensor({{1, 2, 3}, {4, 5, 6}},
    {dtype = torch.float64})
  local parts = matrix:chunk(2, 2)
  assert_eq(#parts, 2)
  assert_eq(parts[1]:totable(), {{1, 2}, {4, 5}})
  assert_eq(parts[2]:totable(), {{3}, {6}})
  assert_eq(#matrix:chunk(4, 2), 3)
  assert_eq(torch.chunk(matrix, 2, 1)[1]:totable(), {{1, 2, 3}})
  assert_error(function() matrix:chunk(0) end, "positive")
  assert_error(function() matrix:chunk(1.5) end, "positive")
  assert_error(function() torch.chunk(1, 2) end, "expects a tensor")
end)

test("unbind removes a dimension and returns its slices", function()
  local matrix = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
  local rows = matrix:unbind(1)
  assert_eq(#rows, 2)
  assert_shape(rows[1], {2})
  assert_eq(rows[1]:totable(), {1, 2})
  assert_eq(rows[2]:totable(), {3, 4})
  local columns = matrix:unbind(2)
  assert_eq(columns[1]:totable(), {1, 3})
  assert_eq(torch.unbind(matrix, -1)[2]:totable(), {2, 4})
  assert_error(function() matrix:unbind(3) end, "dimension")
  assert_error(function() torch.unbind(1) end, "expects a tensor")
end)

test("split and unbind slices remain differentiable and share views", function()
  local matrix = torch.tensor({{1, 2, 3}, {4, 5, 6}},
    {dtype = torch.float64, requires_grad = true})
  matrix:split(1, 2)[1]:sum():backward()
  assert_eq(matrix:grad():totable(), {{1, 0, 0}, {1, 0, 0}})

  local rows = torch.tensor({{1, 2}, {3, 4}},
    {dtype = torch.float64, requires_grad = true})
  rows:unbind(1)[1]:sum():backward()
  assert_eq(rows:grad():totable(), {{1, 1}, {0, 0}})
end)

test("swapaxes transposes the named dimensions", function()
  local matrix = torch.tensor({{1, 2, 3}, {4, 5, 6}},
    {dtype = torch.float64})
  local swapped = matrix:swapaxes(1, 2)
  assert_shape(swapped, {3, 2})
  assert_eq(swapped:totable(), {{1, 4}, {2, 5}, {3, 6}})
  assert_eq(torch.swapaxes(matrix, -1, 1):totable(), swapped:totable())
  assert_error(function() matrix:swapaxes(0, 1) end, "dimension")
  assert_error(function() torch.swapaxes(1, 1, 2) end, "expects a tensor")
end)

test("moveaxis moves one or several axes and preserves the rest", function()
  local cube = torch.arange(24, {dtype = torch.float64}):reshape(2, 3, 4)
  assert_shape(cube:moveaxis(1, 3), {3, 4, 2})
  assert_shape(cube:moveaxis(-1, 1), {4, 2, 3})
  assert_shape(cube:moveaxis({1, 2}, {3, 1}), {4, 3, 2})
  assert_shape(torch.moveaxis(cube, 1, 3), {3, 4, 2})
  assert_shape(torch.movedim(cube, 1, 3), {3, 4, 2})
  assert_error(function() cube:moveaxis({1, 2}, {3}) end, "same length")
  assert_error(function() cube:moveaxis(1.5, 1) end, "integer")
  assert_error(function() torch.moveaxis(1, 1, 2) end, "expects a tensor")
end)

test("moveaxis is differentiable", function()
  local matrix = torch.tensor({{1, 2}, {3, 4}},
    {dtype = torch.float64, requires_grad = true})
  matrix:moveaxis(1, 2):sum():backward()
  assert_eq(matrix:grad():totable(), {{1, 1}, {1, 1}})
end)

test("atleast_1d/2d/3d promote ranks without changing larger inputs", function()
  local scalar = torch.tensor(5, {dtype = torch.float64})
  local vector = torch.tensor({1, 2}, {dtype = torch.float64})
  local matrix = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})

  assert_shape(torch.atleast_1d(scalar), {1})
  assert_shape(torch.atleast_1d(vector), {2})
  local a, b = torch.atleast_2d(scalar, vector)
  assert_shape(a, {1, 1})
  assert_shape(b, {1, 2})
  local c, d, e = torch.atleast_3d(scalar, vector, matrix)
  assert_shape(c, {1, 1, 1})
  assert_shape(d, {1, 2, 1})
  assert_shape(e, {2, 2, 1})

  local cube = torch.tensor({{{1}}}, {dtype = torch.float64})
  local same = torch.atleast_3d(cube)
  assert_true(same == cube)
end)

test("atleast promotions are differentiable and validate tensors", function()
  local scalar = torch.tensor(5, {dtype = torch.float64,
    requires_grad = true})
  torch.atleast_2d(scalar):sum():backward()
  assert_eq(scalar:grad():item(), 1)

  assert_error(function() torch.atleast_1d(1) end, "expects tensors")
  assert_error(function() torch.atleast_3d({}) end, "expects tensors")
end)
