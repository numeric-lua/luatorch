-- Sequence utilities: argsort, diff, roll, tile, and repeat_interleave.

local torch = require("torch")

test("argsort returns 1-based sort indices", function()
  local matrix = torch.tensor({{3, 1, 2}, {6, 5, 4}},
    {dtype = torch.float64})
  local indices = matrix:argsort()
  assert_eq(indices:dtype(), "int64")
  assert_eq(indices:totable(), {{2, 3, 1}, {3, 2, 1}})
  assert_eq(matrix:argsort(1):totable(), {{1, 1, 1}, {2, 2, 2}})
  assert_eq(matrix:argsort(2, true):totable(), {{1, 3, 2}, {1, 2, 3}})
  assert_eq(torch.argsort(matrix):totable(), indices:totable())
end)

test("diff computes first and higher-order differences", function()
  local values = torch.tensor({1, 3, 6, 10}, {dtype = torch.float64})
  assert_eq(values:diff():totable(), {2, 3, 4})
  assert_eq(values:diff(2):totable(), {1, 1})
  assert_eq(torch.diff(values, 2):totable(), {1, 1})
  local matrix = torch.tensor({{1, 10, 100}, {2, 20, 200}},
    {dtype = torch.float64})
  assert_eq(matrix:diff(1, 2):totable(), {{9, 90}, {18, 180}})
  assert_eq(matrix:diff(1, 1):totable(), {{1, 10, 100}})
end)

test("diff supports n=0, empty dimensions, and autograd", function()
  local values = torch.tensor({1, 2}, {dtype = torch.float64})
  local clone = values:diff(0)
  assert_eq(clone:totable(), {1, 2})
  assert_false(clone == values)
  assert_shape(torch.tensor({}, {dtype = torch.float64}):diff(), {0})
  assert_shape(torch.tensor({5}, {dtype = torch.float64}):diff(), {0})

  local grad_values = torch.tensor({1, 3, 6},
    {dtype = torch.float64, requires_grad = true})
  grad_values:diff():sum():backward()
  assert_eq(grad_values:grad():totable(), {-1, 0, 1})
end)

test("diff validates n and dtype", function()
  local values = torch.tensor({1, 2, 3})
  assert_error(function() values:diff(-1) end, "non-negative")
  assert_error(function() values:diff(1.5) end, "integer")
  assert_error(function() torch.tensor({true, false}):diff() end, "bool")
  assert_error(function() torch.diff(1) end, "expects a tensor")
end)

test("roll shifts flattened tensors and individual dimensions", function()
  local values = torch.tensor({1, 2, 3, 4}, {dtype = torch.float64})
  assert_eq(values:roll(1):totable(), {4, 1, 2, 3})
  assert_eq(values:roll(-1):totable(), {2, 3, 4, 1})
  assert_eq(torch.roll(values, 2):totable(), {3, 4, 1, 2})

  local matrix = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
  assert_eq(matrix:roll(1, 1):totable(), {{3, 4}, {1, 2}})
  assert_eq(matrix:roll(1, 2):totable(), {{2, 1}, {4, 3}})
  assert_eq(matrix:roll({1, 1}, {1, 2}):totable(), {{4, 3}, {2, 1}})
  assert_eq(matrix:roll(1):totable(), {{4, 1}, {2, 3}})
end)

test("roll is differentiable, handles empty tensors, and validates arguments", function()
  local values = torch.tensor({1, 2, 3},
    {dtype = torch.float64, requires_grad = true})
  values:roll(1):sum():backward()
  assert_eq(values:grad():totable(), {1, 1, 1})

  local empty = torch.zeros(0, {dtype = torch.float64})
  assert_shape(empty:roll(1), {0})
  assert_shape(empty:roll(1, 1), {0})

  local matrix = torch.tensor({{1, 2}, {3, 4}})
  assert_error(function() matrix:roll({1, 2}, {1}) end, "same length")
  assert_error(function() matrix:roll(1, 1.5) end, "integer")
  assert_error(function() matrix:roll(1.5) end, "integer")
  assert_error(function() torch.roll(1, 1) end, "expects a tensor")
end)

test("tile repeats tensors and promotes ranks", function()
  local vector = torch.tensor({1, 2}, {dtype = torch.float64})
  assert_eq(vector:tile({2}):totable(), {1, 2, 1, 2})
  assert_eq(vector:tile({2, 3}):totable(),
    {{1, 2, 1, 2, 1, 2}, {1, 2, 1, 2, 1, 2}})
  assert_eq(torch.tile(vector, {1, 2}):totable(),
    {{1, 2, 1, 2}})

  local matrix = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
  assert_eq(matrix:tile({2, 1}):totable(),
    {{1, 2}, {3, 4}, {1, 2}, {3, 4}})
  assert_eq(torch.tensor(5, {dtype = torch.float64}):tile({2, 3})
    :totable(), {{5, 5, 5}, {5, 5, 5}})
  assert_shape(vector:tile({0}), {0})
end)

test("tile is differentiable and validates repeats", function()
  local values = torch.tensor({1, 2},
    {dtype = torch.float64, requires_grad = true})
  values:tile({2}):sum():backward()
  assert_eq(values:grad():totable(), {2, 2})

  local vector = torch.tensor({1, 2})
  assert_error(function() vector:tile({}) end, "non-empty")
  assert_error(function() vector:tile({1.5}) end, "non-negative")
  assert_error(function() vector:tile({-1}) end, "non-negative")
  assert_error(function() torch.tile(1, {2}) end, "expects a tensor")
end)

test("repeat_interleave repeats elements and dimensions", function()
  local vector = torch.tensor({1, 2, 3}, {dtype = torch.float64})
  assert_eq(vector:repeat_interleave(2):totable(), {1, 1, 2, 2, 3, 3})
  assert_eq(vector:repeat_interleave(torch.tensor({1, 2, 0},
    {dtype = torch.int64})):totable(), {1, 2, 2})
  assert_eq(torch.repeat_interleave(vector, 2):totable(),
    {1, 1, 2, 2, 3, 3})
  assert_shape(vector:repeat_interleave(0), {0})

  local matrix = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
  assert_eq(matrix:repeat_interleave(2, 1):totable(),
    {{1, 2}, {1, 2}, {3, 4}, {3, 4}})
  assert_eq(matrix:repeat_interleave(2, 2):totable(),
    {{1, 1, 2, 2}, {3, 3, 4, 4}})
  assert_eq(matrix:repeat_interleave(2):totable(),
    {1, 1, 2, 2, 3, 3, 4, 4})
end)

test("repeat_interleave is differentiable and validates inputs", function()
  local values = torch.tensor({1, 2},
    {dtype = torch.float64, requires_grad = true})
  values:repeat_interleave(2):sum():backward()
  assert_eq(values:grad():totable(), {2, 2})

  local vector = torch.tensor({1, 2})
  assert_error(function() vector:repeat_interleave(1.5) end, "integer")
  assert_error(function() vector:repeat_interleave(-1) end, "non-negative")
  assert_error(function() vector:repeat_interleave(torch.tensor({1},
    {dtype = torch.int64})) end, "matching")
  assert_error(function() vector:repeat_interleave(torch.tensor({1.0, 2.0}))
  end, "integer")
  assert_error(function() torch.repeat_interleave(1, 2) end,
    "expects a tensor")
end)
