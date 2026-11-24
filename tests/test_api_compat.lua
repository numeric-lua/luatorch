-- PyTorch API-parity helpers: aminmax, count_nonzero, isclose/allclose with
-- equal_nan, nan_to_num, the torch.linalg matmul-family aliases, and the
-- NaN-payload read fix they exercise.

local torch = require("torch")

test("aminmax returns the minimum and maximum over all elements", function()
  local t = torch.tensor({{3, 1}, {4, -2}}, {dtype = torch.float64})
  local lo, hi = torch.aminmax(t)
  assert_eq(lo:item(), -2)
  assert_eq(hi:item(), 4)
  assert_eq(lo:sizes(), {})
  assert_eq(hi:sizes(), {})
  assert_eq(lo:dtype(), "float64")
  assert_eq(hi:dtype(), "float64")
  local lo2, hi2 = t:aminmax()
  assert_eq(lo2:item(), -2)
  assert_eq(hi2:item(), 4)
end)

test("aminmax reduces dimensions, keeps them, and propagates NaN", function()
  local t = torch.tensor({{3, 1, 4}, {5, 2, 6}}, {dtype = torch.float64})
  local lo, hi = t:aminmax(2)
  assert_eq(lo:sizes(), {2})
  assert_eq(lo:totable(), {1, 2})
  assert_eq(hi:totable(), {4, 6})
  local lo2, hi2 = t:aminmax(1, true)
  assert_eq(lo2:sizes(), {1, 3})
  assert_eq(lo2:totable(), {{3, 1, 4}})
  assert_eq(hi2:totable(), {{5, 2, 6}})
  local lo3, hi3 = t:aminmax(-1)
  assert_eq(lo3:totable(), {1, 2})
  assert_eq(hi3:totable(), {4, 6})

  local with_nan = torch.tensor({1, 0 / 0, 3}, {dtype = torch.float64})
  local mn, mx = with_nan:aminmax()
  assert_true(mn:item() ~= mn:item(), "aminmax minimum should be NaN")
  assert_true(mx:item() ~= mx:item(), "aminmax maximum should be NaN")
  local row_mn, row_mx = with_nan:reshape(1, 3):aminmax(2)
  assert_true(row_mn:get(1) ~= row_mn:get(1))
  assert_true(row_mx:get(1) ~= row_mx:get(1))
end)

test("aminmax validates its dimension and keepdim arguments", function()
  local t = torch.tensor({{1, 2}, {3, 4}})
  assert_error(function() t:aminmax(1.5) end, "dim")
  assert_error(function() t:aminmax("1") end, "dim")
  assert_error(function() t:aminmax(0) end, "dimension 0")
  assert_error(function() t:aminmax(1, 1) end, "keepdim")
end)

test("count_nonzero counts across the tensor and dimensions", function()
  local t = torch.tensor({{1, 0, 3}, {0, 5, 0}}, {dtype = torch.float64})
  local total = t:count_nonzero()
  assert_eq(total:item(), 3)
  assert_eq(total:dtype(), "int64")
  assert_eq(total:sizes(), {})
  assert_eq(t:count_nonzero(1):totable(), {1, 1, 1})
  assert_eq(t:count_nonzero(2):totable(), {2, 1})
  assert_eq(t:count_nonzero(-1):totable(), {2, 1})
  assert_eq(t:count_nonzero({1, 2}):item(), 3)

  local cube = torch.tensor(
    {{{1, 0}, {0, 0}, {0, 0}}, {{0, 0}, {1, 1}, {0, 0}}},
    {dtype = torch.float64})
  assert_eq(cube:count_nonzero({1, 3}):totable(), {1, 2, 0})

  local flags = torch.tensor({true, false, true, true})
  assert_eq(flags:count_nonzero():item(), 3)
  assert_eq(torch.tensor(0):count_nonzero():item(), 0)
  assert_eq(torch.tensor(5):count_nonzero():item(), 1)
  local mixed = torch.tensor({0 / 0, 0, 2}, {dtype = torch.float64})
  assert_eq(mixed:count_nonzero():item(), 2)
end)

test("count_nonzero validates its dimension argument", function()
  local t = torch.tensor({{1, 2}, {3, 4}})
  assert_error(function() t:count_nonzero("1") end, "dim")
  assert_error(function() t:count_nonzero(1.5) end, "integer")
  assert_error(function() t:count_nonzero(0) end, "out of range")
  assert_error(function() t:count_nonzero(3) end, "out of range")
  assert_error(function() t:count_nonzero({}) end, "empty")
  assert_error(function() t:count_nonzero({1, 1}) end, "repeat")
  assert_error(function() t:count_nonzero({3}) end, "out of range")
  assert_error(function() t:count_nonzero({1.5}) end, "integers")
  assert_error(function() torch.count_nonzero(1) end, "expects a tensor")
end)

test("isclose compares values elementwise with tolerances", function()
  local a = torch.tensor({1, 2, 3}, {dtype = torch.float64})
  local b = torch.tensor({1, 2.0000001, 3.1}, {dtype = torch.float64})
  local close = a:isclose(b)
  assert_eq(close:dtype(), "bool")
  assert_eq(close:totable(), {true, true, false})
  assert_eq(a:isclose(a):totable(), {true, true, true})
  -- The tolerance is |a - b| <= atol + rtol * |b|.
  assert_false(a:isclose(b, 0, 0):get(2))
  assert_true(a:isclose(b, 0, 1e-6):get(2))

  local rows = torch.tensor({{1, 2}, {3, 4}}, {dtype = torch.float64})
  local row = torch.tensor({1, 4}, {dtype = torch.float64})
  assert_eq(rows:isclose(row):totable(), {{true, false}, {false, true}})

  local integers = torch.tensor({1, 2}, {dtype = torch.int64})
  assert_eq(integers:isclose(torch.tensor({1, 3},
    {dtype = torch.int64})):totable(), {true, false})
end)

test("isclose treats matching infinities and NaNs correctly", function()
  local inf = math.huge
  local values = torch.tensor({inf, -inf, 0 / 0, 2}, {dtype = torch.float64})
  local same = torch.tensor({inf, -inf, 0 / 0, 2}, {dtype = torch.float64})
  assert_eq(values:isclose(same):totable(), {true, true, false, true})
  assert_eq(values:isclose(same, nil, nil, true):totable(),
    {true, true, true, true})
  assert_false(values:isclose(torch.tensor({-inf, inf, 1, 2},
    {dtype = torch.float64})):get(1))
  assert_false(values:isclose(torch.tensor({1, -inf, 1, 2},
    {dtype = torch.float64})):get(1))
end)

test("allclose accepts equal_nan and keeps its old defaults", function()
  local a = torch.tensor({1, 0 / 0, 3}, {dtype = torch.float64})
  local b = torch.tensor({1, 0 / 0, 3}, {dtype = torch.float64})
  assert_false(a:allclose(b))
  assert_true(a:allclose(b, nil, nil, true))
  assert_true(torch.allclose(a, b, nil, nil, true))
  assert_false(torch.allclose(torch.tensor({1}), torch.tensor({2})))
  assert_true(torch.allclose(torch.tensor({1}), torch.tensor({1})))
end)

test("isclose validates its arguments", function()
  local a = torch.tensor({1})
  assert_error(function() a:isclose(3) end, "expects a tensor")
  assert_error(function() a:isclose(torch.tensor({true})) end, "bool")
  assert_error(function() a:isclose(a, -1) end, "rtol")
  assert_error(function() a:isclose(a, nil, -1) end, "atol")
  assert_error(function() a:isclose(a, nil, nil, 1) end, "equal_nan")
  assert_error(function() torch.isclose(1, a) end, "expects a tensor")
end)

test("nan_to_num replaces NaN and infinities with dtype defaults", function()
  local t = torch.tensor({1, 0 / 0, math.huge, -math.huge},
    {dtype = torch.float64})
  local out = t:nan_to_num()
  assert_eq(out:dtype(), "float64")
  assert_eq(out:totable(), {1, 0, 1.7976931348623157e308,
    -1.7976931348623157e308})
  -- The input is unchanged (nan_to_num is out-of-place).
  assert_true(t:get(2) ~= t:get(2))

  local f32 = torch.tensor({0 / 0, math.huge, -math.huge},
    {dtype = torch.float32}):nan_to_num()
  assert_eq(f32:totable(), {0, 3.4028234663852886e38,
    -3.4028234663852886e38})
  local f16 = torch.tensor({0 / 0, math.huge, -math.huge},
    {dtype = torch.float16}):nan_to_num()
  assert_eq(f16:totable(), {0, 65504, -65504})

  local custom = t:nan_to_num(-1, 7, -7)
  assert_eq(custom:totable(), {1, -1, 7, -7})
  assert_eq(torch.nan_to_num(torch.tensor(5, {dtype = torch.float64}))
    :item(), 5)
end)

test("nan_to_num keeps gradients for finite entries only", function()
  local x = torch.tensor({1, 0 / 0, 3, math.huge},
    {dtype = torch.float64, requires_grad = true})
  local y = x:nan_to_num()
  assert_true(y:requires_grad())
  y:sum():backward()
  assert_eq(x:grad():totable(), {1, 0, 1, 0})

  local plain = torch.tensor({1, 2, 3}, {dtype = torch.float64,
    requires_grad = true})
  plain:nan_to_num():sum():backward()
  assert_eq(plain:grad():totable(), {1, 1, 1})
end)

test("nan_to_num leaves integer and bool tensors unchanged", function()
  local integers = torch.tensor({1, 2}, {dtype = torch.int64})
  local int_out = integers:nan_to_num()
  assert_eq(int_out:dtype(), "int64")
  assert_eq(int_out:totable(), {1, 2})
  assert_false(int_out == integers)

  local flags = torch.tensor({true, false})
  local bool_out = flags:nan_to_num()
  assert_eq(bool_out:dtype(), "bool")
  assert_eq(bool_out:totable(), {true, false})
  assert_error(function() integers:nan_to_num("x") end, "number")
  assert_error(function() torch.nan_to_num(1) end, "expects a tensor")
end)

test("linalg matmul, mm, mv, and outer aliases match tensor methods", function()
  local a = torch.tensor({{1, 2}, {3, 4}},
    {dtype = torch.float64, requires_grad = true})
  local b = torch.tensor({{5, 6}, {7, 8}}, {dtype = torch.float64})
  assert_true(torch.linalg.matmul(a, b):allclose(a:matmul(b)))
  assert_true(torch.linalg.mm(a, b):allclose(a:mm(b)))

  local v = torch.tensor({1, 2}, {dtype = torch.float64})
  assert_true(torch.linalg.mv(a, v):allclose(a:mv(v)))

  local x = torch.tensor({1, 2}, {dtype = torch.float64})
  local y = torch.tensor({3, 4}, {dtype = torch.float64})
  assert_true(torch.linalg.outer(x, y):allclose(x:outer(y)))
  assert_eq(torch.linalg.outer(x, y):totable(), {{3, 4}, {6, 8}})

  local product = torch.linalg.matmul(a, b)
  product:sum():backward()
  assert_true(a:grad():allclose(torch.tensor({{11, 15}, {11, 15}},
    {dtype = torch.float64})))

  assert_error(function() torch.linalg.matmul(1, b) end, "two tensors")
  assert_error(function() torch.linalg.mm(a, 1) end, "two tensors")
  assert_error(function() torch.linalg.mv(1, v) end, "two tensors")
  assert_error(function() torch.linalg.outer(x, 1) end, "two tensors")
end)

test("NaN payloads round-trip through item, totable, and iter", function()
  for _, dtype in ipairs({ torch.float16, torch.float32, torch.float64 }) do
    local t = torch.tensor({1, 0 / 0, 3}, {dtype = dtype})
    -- min/max reductions can produce a NaN payload that collides with
    -- LuaJIT's NaN boxing; reading it must still yield a Lua NaN.
    assert_true(t:min():item() ~= t:min():item())
    assert_true(t:max():item() ~= t:max():item())
    local mn, mx = t:aminmax()
    assert_true(mn:item() ~= mn:item())
    assert_true(mx:item() ~= mx:item())

    local values = t:totable()
    assert_true(values[2] ~= values[2])
    local count, nans = 0, 0
    for value in t:iter() do
      count = count + 1
      if value ~= value then nans = nans + 1 end
    end
    assert_eq(count, 3)
    assert_eq(nans, 1)
  end
end)
