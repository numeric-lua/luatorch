-- torch.mode, nanmedian/nanquantile, torch.special, and nn.functional
-- activation wrappers.

local torch = require("torch")
local F = require("torch.nn.functional")

test("mode returns the most frequent value and 1-based index", function()
  local values = torch.tensor({1, 2, 2, 3, 3, 3}, {dtype = torch.float64})
  local mode, indices = values:mode()
  assert_eq(mode:item(), 3)
  assert_eq(indices:item(), 6)

  local matrix = torch.tensor({{1, 2, 2}, {3, 3, 1}},
    {dtype = torch.float64})
  local values2, indices2 = matrix:mode(2)
  assert_eq(values2:totable(), {2, 3})
  assert_eq(indices2:totable(), {3, 2})
  local values1, indices1 = matrix:mode(1)
  assert_eq(values1:totable(), {1, 2, 1})
  assert_eq(indices1:totable(), {1, 1, 2})
  local alias_mode = torch.mode(values)
  assert_eq(alias_mode:item(), 3)
  assert_error(function() torch.mode(1) end, "expects a tensor")
end)

test("nanmedian ignores NaNs with and without a dimension", function()
  local nan = 0 / 0
  local values = torch.tensor({1, nan, 2, 3}, {dtype = torch.float64})
  assert_eq(values:nanmedian():item(), 2)

  local matrix = torch.tensor({{1, nan, 3}, {4, 5, nan}},
    {dtype = torch.float64})
  local medians, indices = matrix:nanmedian(2)
  assert_eq(medians:totable(), {1, 4})
  assert_eq(indices:totable(), {1, 1})
  local by_column = matrix:nanmedian(1)
  assert_eq(by_column:totable(), {1, 5, 3})
  assert_eq(torch.nanmedian(values):item(), 2)
  assert_error(function() torch.nanmedian(1) end, "expects a tensor")
end)

test("nanquantile ignores NaNs and interpolates", function()
  local nan = 0 / 0
  local values = torch.tensor({1, nan, 2, 3}, {dtype = torch.float64})
  assert_eq(values:nanquantile(0.5):item(), 2)
  assert_eq(values:nanquantile(0.25):item(), 1.5)
  assert_eq(torch.nanquantile(values, 0.75):item(), 2.5)

  local matrix = torch.tensor({{1, nan, 3}, {4, 5, nan}},
    {dtype = torch.float64})
  assert_eq(matrix:nanquantile(0.5, 2):totable(), {2, 4.5})
  assert_shape(matrix:nanquantile(0.5, 2, true), {2, 1})

  assert_error(function() values:nanquantile(-0.1) end, "in [0, 1]")
  assert_error(function() values:nanquantile(1.1) end, "in [0, 1]")
  assert_error(function() torch.nanquantile(1, 0.5) end,
    "expects a tensor")
end)

test("torch.special exposes the special-function namespace", function()
  local values = torch.tensor({0, 1}, {dtype = torch.float64})
  assert_eq(type(torch.special), "table")
  assert_eq(torch.special.ndtr(values):get(1), 0.5)
  assert_near(torch.special.psi(values):get(2), -0.5772156649015329, 1e-12)
  assert_eq(torch.special.expit(values):get(2),
    torch.sigmoid(values):get(2))
  assert_eq(torch.special.logit(torch.tensor({0.5},
    {dtype = torch.float64})):get(1), 0)
  assert_eq(torch.special.erf(values):get(2), torch.erf(values):get(2))
  assert_eq(torch.special.entr(values):get(2), torch.entr(values):get(2))
  assert_eq(torch.special.xlogy(values, values):get(2),
    torch.xlogy(values, values):get(2))
  assert_eq(torch.special.xlog1py(values, values):get(2),
    torch.xlog1py(values, values):get(2))
end)

test("nn.functional exposes the new activation wrappers", function()
  local values = torch.tensor({-1, 0, 1}, {dtype = torch.float64})
  assert_near(F.softplus(values):get(2), math.log(2), 1e-12)
  assert_near(F.elu(values):get(1), math.exp(-1) - 1, 1e-12)
  assert_near(F.selu(values):get(1),
    1.0507009873554805 * 1.6732632423543772 * (math.exp(-1) - 1), 1e-12)
  assert_near(F.silu(values):get(3), 1 / (1 + math.exp(-1)), 1e-12)
  assert_near(F.gelu(values):get(3), 0.8413447460685429, 1e-12)
  assert_eq(F.hardsigmoid(values):totable(), {1 / 3, 0.5, 2 / 3})
  assert_near(F.hardswish(values):get(1), -1 / 3, 1e-12)
  assert_near(F.softsign(torch.tensor({3}, {dtype = torch.float64}))
    :get(1), 0.75, 1e-12)
end)

test("nn.functional in-place activation wrappers mutating their input", function()
  local input = torch.tensor({-1}, {dtype = torch.float64})
  local result = F.elu(input, 1, true)
  assert_true(result == input)
  assert_near(input:get(1), math.exp(-1) - 1, 1e-12)

  local sigmoid_input = torch.tensor({0}, {dtype = torch.float64})
  F.hardsigmoid(sigmoid_input, true)
  assert_eq(sigmoid_input:get(1), 0.5)
end)
