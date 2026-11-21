local torch = require("torch")

local function float_matrix(rows)
  return torch.tensor(rows, { dtype = torch.float64 })
end

test("det and inverse", function()
  local matrix = float_matrix({ { 4, 7 }, { 2, 6 } })
  assert_near(matrix:det():item(), 10, 1e-10)
  assert_near(matrix:inverse():totable(), {
    { 0.6, -0.7 },
    { -0.2, 0.4 },
  }, 1e-10)

  local identity = matrix:matmul(matrix:inverse())
  assert_near(identity:totable(), { { 1, 0 }, { 0, 1 } }, 1e-10)
end)

test("linalg aliases", function()
  local matrix = float_matrix({ { 4, 7 }, { 2, 6 } })
  assert_near(torch.linalg.det(matrix):item(), 10, 1e-10)
  assert_near(torch.linalg.inv(matrix):get(1, 1), 0.6, 1e-10)
end)

test("solve", function()
  local a = float_matrix({ { 3, 1 }, { 1, 2 } })
  local b = torch.tensor({ 9, 8 }, { dtype = torch.float64 })
  assert_near(a:solve(b):totable(), { 2, 3 }, 1e-10)

  local multiple = float_matrix({ { 9, 5 }, { 8, 5 } })
  assert_near(a:solve(multiple):totable(), {
    { 2, 1 },
    { 3, 2 },
  }, 1e-9)
end)

test("cholesky", function()
  local matrix = float_matrix({ { 4, 2 }, { 2, 3 } })
  local lower = matrix:cholesky()
  assert_near(lower:matmul(lower:t()):totable(), matrix:totable(), 1e-9)

  local upper = matrix:cholesky(true)
  assert_near(upper:t():matmul(upper):totable(), matrix:totable(), 1e-9)
end)

test("svd reconstruction", function()
  local matrix = float_matrix({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local u, s, vh = matrix:svd(false)
  assert_shape(u, { 3, 2 })
  assert_shape(s, { 2 })
  assert_shape(vh, { 2, 2 })

  local diagonal = torch.zeros(2, 2, { dtype = torch.float64 })
  for i = 1, 2 do
    diagonal:set(i, i, s:get(i))
  end
  assert_near(u:matmul(diagonal):matmul(vh):totable(), matrix:totable(), 1e-9)
end)

test("qr reconstruction", function()
  local matrix = float_matrix({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local q, r = matrix:qr()
  assert_shape(q, { 3, 2 })
  assert_shape(r, { 2, 2 })
  assert_near(q:matmul(r):totable(), matrix:totable(), 1e-9)

  local gram = q:t():matmul(q)
  assert_near(gram:totable(), { { 1, 0 }, { 0, 1 } }, 1e-9)
end)

test("mv and outer", function()
  local matrix = torch.tensor({ { 1, 2, 3 }, { 4, 5, 6 } })
  local vector = torch.tensor({ 1, 1, 1 })
  assert_tensor(matrix:mv(vector), { 6, 15 })

  assert_tensor(torch.tensor({ 1, 2 }):outer(torch.tensor({ 3, 4, 5 })), {
    { 3, 4, 5 },
    { 6, 8, 10 },
  })
end)

test("addmm and baddbmm", function()
  local mat1 = torch.ones(2, 3)
  local mat2 = torch.ones(3, 2)
  local input = torch.ones(2, 2)
  assert_tensor(input:addmm(mat1, mat2), { { 4, 4 }, { 4, 4 } })
  assert_tensor(input:addmm(mat1, mat2, 2, 3), { { 11, 11 }, { 11, 11 } })

  local batch_input = torch.zeros(2, 2, 2)
  local batch1 = torch.ones(2, 2, 3)
  local batch2 = torch.ones(2, 3, 2)
  assert_tensor(batch_input:baddbmm(batch1, batch2), {
    { { 3, 3 }, { 3, 3 } },
    { { 3, 3 }, { 3, 3 } },
  })
end)

test("linalg errors", function()
  assert_error(function()
    torch.zeros(2, 3):inverse()
  end)
  assert_error(function()
    torch.zeros(2, 3):det()
  end)
  assert_error(function()
    torch.ones(2, 2):cholesky()
  end)
end)

test("slogdet sign and logabsdet", function()
  local positive = float_matrix({ { 4, 7 }, { 2, 6 } })
  local sign, logabsdet = positive:slogdet()
  assert_dtype(sign, "float64")
  assert_dtype(logabsdet, "float64")
  assert_eq(sign:item(), 1)
  assert_near(logabsdet:item(), math.log(10), 1e-12)
  assert_near((sign * logabsdet:exp()):item(), positive:det():item(), 1e-10)

  local negative = float_matrix({ { 1, 2 }, { 3, 4 } })
  local negative_sign, negative_logabsdet = torch.linalg.slogdet(negative)
  assert_eq(negative_sign:item(), -1)
  assert_near(negative_logabsdet:item(), math.log(2), 1e-12)
  assert_near((negative_sign * negative_logabsdet:exp()):item(),
    negative:det():item(), 1e-12)
end)

test("slogdet handles batches and empty matrices", function()
  local batch = torch.randn(3, 2, 2, { dtype = torch.float64 })
  local sign, logabsdet = batch:slogdet()
  assert_shape(sign, { 3 })
  assert_shape(logabsdet, { 3 })
  assert_near((sign * logabsdet:exp()):totable(), batch:det():totable(), 1e-9)

  local empty_sign, empty_logabsdet = torch.zeros(0, 0,
    { dtype = torch.float64 }):slogdet()
  assert_eq(empty_sign:item(), 1)
  assert_eq(empty_logabsdet:item(), 0)
end)

test("slogdet gradients flow through logabsdet", function()
  local matrix = torch.tensor({ { 2, 1 }, { 1, 3 } },
    { dtype = torch.float64, requires_grad = true })
  local sign, logabsdet = matrix:slogdet()
  assert_true(sign:requires_grad())
  assert_true(logabsdet:requires_grad())
  logabsdet:backward()
  -- d log|det A| / dA = A^{-T}; A^{-1} = [[3, -1], [-1, 2]] / 5.
  assert_near(matrix:grad():totable(), {
    { 0.6, -0.2 },
    { -0.2, 0.4 },
  }, 1e-10)
end)

test("slogdet rejects non-square matrices", function()
  assert_error(function()
    torch.zeros(2, 3):slogdet()
  end, "square")
end)

test("lstsq solves overdetermined systems", function()
  -- Inconsistent system; the normal equations give x = (-2/3, 2/3) and a
  -- residual sum of squares of 2/3.
  local a = float_matrix({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local b = torch.tensor({ 1, 0, 1 }, { dtype = torch.float64 })
  local solution, residuals, rank, singular_values =
    a:lstsq(b, { driver = "gelsd" })

  assert_shape(solution, { 2 })
  assert_near(solution:totable(), { -2 / 3, 2 / 3 }, 1e-9)
  assert_shape(residuals, { 1 })
  assert_near(residuals:get(1), 2 / 3, 1e-9)
  assert_eq(rank:dim(), 0)
  assert_eq(rank:item(), 2)
  assert_dtype(rank, "int64")
  assert_shape(singular_values, { 2 })

  local normal = a:t():mm(a):mv(solution)
  assert_near(normal:totable(), a:t():mv(b):totable(), 1e-9)

  assert_tensor(torch.linalg.lstsq(a, b, { driver = "gelsd" }),
    solution:totable(), 1e-9)
end)

test("lstsq supports multiple right-hand sides and drivers", function()
  local a = float_matrix({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local b = float_matrix({ { 1, 1 }, { 2, 0 }, { 3, 1 } })
  local solution, residuals, rank, singular_values =
    a:lstsq(b, { driver = "gelsd" })

  assert_shape(solution, { 2, 2 })
  assert_tensor(solution, { { 0, -2 / 3 }, { 0.5, 2 / 3 } }, 1e-9)
  assert_shape(residuals, { 2 })
  assert_tensor(residuals, { 0, 2 / 3 }, 1e-9)
  assert_eq(rank:item(), 2)
  assert_shape(singular_values, { 2 })

  -- `gels` assumes full rank and returns empty rank/singular values.
  local gels_solution = a:lstsq(b, { driver = "gels" })
  assert_tensor(gels_solution, solution:totable(), 1e-8)

  local gelss_solution, _, gelss_rank, gelss_values =
    a:lstsq(b, { driver = "gelss" })
  assert_tensor(gelss_solution, solution:totable(), 1e-8)
  assert_eq(gelss_rank:item(), 2)
  assert_shape(gelss_values, { 2 })

  -- The default driver is ATen's CPU default (`gelsy`).
  assert_tensor(a:lstsq(b), solution:totable(), 1e-8)
end)

test("lstsq handles batches, empty residuals, and wide systems", function()
  local a = torch.tensor({
    { { 1, 0 }, { 0, 1 }, { 1, 1 } },
    { { 1, 0 }, { 0, 1 }, { 1, 1 } },
  }, { dtype = torch.float64 })
  local b = float_matrix({ { 1, 2, 3 }, { 2, 1, 3 } })
  local solution, residuals, rank, singular_values =
    a:lstsq(b, { driver = "gelsd" })

  assert_shape(solution, { 2, 2 })
  assert_tensor(solution, { { 1, 2 }, { 2, 1 } }, 1e-9)
  assert_shape(residuals, { 2, 1 })
  assert_tensor(residuals, { { 0 }, { 0 } }, 1e-9)
  assert_eq(rank:dim(), 1)
  assert_tensor(rank, { 2, 2 })
  assert_shape(singular_values, { 2, 2 })

  -- Underdetermined systems return the minimum-norm solution and no
  -- residuals.
  local wide = float_matrix({ { 1, 2, 3 }, { 4, 5, 6 } })
  local wide_b = torch.tensor({ 1, 2 }, { dtype = torch.float64 })
  local wide_solution, wide_residuals, wide_rank, wide_values =
    wide:lstsq(wide_b, { driver = "gelsd" })

  assert_shape(wide_solution, { 3 })
  assert_near(wide_solution:totable(), { -1 / 18, 1 / 9, 5 / 18 }, 1e-9)
  assert_eq(wide_residuals:numel(), 0)
  assert_eq(wide_rank:item(), 2)
  assert_shape(wide_values, { 2 })
  assert_near(wide:mv(wide_solution):totable(), { 1, 2 }, 1e-9)
end)

test("lstsq gradients flow through the solution", function()
  local a = torch.tensor({ { 2, 1 }, { 1, 3 } },
    { dtype = torch.float64, requires_grad = true })
  local b = torch.tensor({ 1, 1 },
    { dtype = torch.float64, requires_grad = true })
  local solution = a:lstsq(b)
  assert_true(solution:requires_grad())
  solution:sum():backward()

  -- x = A^-1 b, so d sum(x) / db = A^-T 1 and
  -- d sum(x) / dA = -(A^-T 1) x^T.
  assert_near(a:grad():totable(), {
    { -0.16, -0.08 },
    { -0.08, -0.04 },
  }, 1e-10)
  assert_near(b:grad():totable(), { 0.4, 0.2 }, 1e-10)
end)

test("lstsq rcond controls the numerical rank", function()
  local rank_deficient = float_matrix({ { 1, 2 }, { 2, 4 } })
  local b = torch.tensor({ 1, 2 }, { dtype = torch.float64 })

  local solution, _, rank =
    rank_deficient:lstsq(b, { driver = "gelss", rcond = 0 })
  assert_eq(rank:item(), 2)
  assert_near(solution:totable(), { 0, 0.5 }, 1e-9)

  local truncated, _, truncated_rank =
    rank_deficient:lstsq(b, { driver = "gelss", rcond = 1e-10 })
  assert_eq(truncated_rank:item(), 1)
  assert_near(truncated:totable(), { 0.2, 0.4 }, 1e-9)

  local truncated_all, _, zero_rank =
    rank_deficient:lstsq(b, { driver = "gelss", rcond = 1 })
  assert_eq(zero_rank:item(), 0)
  assert_near(truncated_all:totable(), { 0, 0 }, 1e-9)
end)

test("lstsq validates options and input shapes", function()
  local a = float_matrix({ { 1, 0 }, { 0, 1 }, { 1, 1 } })
  local b = torch.tensor({ 1, 2, 3 }, { dtype = torch.float64 })

  assert_error(function()
    a:lstsq(b, { driver = "nope" })
  end, "driver must be one of")
  assert_error(function()
    a:lstsq(b, "gels")
  end, "options must be a table")
  assert_error(function()
    a:lstsq(b, { rcond = -1 })
  end, "rcond must be a non-negative number")
  assert_error(function()
    a:lstsq({ 1, 2, 3 })
  end, "right-hand side")
  assert_error(function()
    a:lstsq(torch.ones(2, { dtype = torch.float64 }))
  end, "should match")
  assert_error(function()
    torch.ones(3, 2, { dtype = torch.float16 }):lstsq(
      torch.ones(3, { dtype = torch.float16 }))
  end, "float and double")
end)

test("matrix_power computes integer powers", function()
  local a = float_matrix({ { 1, 2 }, { 3, 4 } })
  assert_tensor(a:matrix_power(0), { { 1, 0 }, { 0, 1 } })
  assert_tensor(a:matrix_power(1), a:totable())
  assert_tensor(a:matrix_power(2), { { 7, 10 }, { 15, 22 } })
  assert_tensor(a:matrix_power(3), { { 37, 54 }, { 81, 118 } })

  -- A^-1 = [[-2, 1], [1.5, -0.5]]
  assert_tensor(a:matrix_power(-1), { { -2, 1 }, { 1.5, -0.5 } }, 1e-12)
  assert_tensor(a:matrix_power(-2),
    { { 5.5, -2.5 }, { -3.75, 1.75 } }, 1e-12)
  assert_tensor(torch.linalg.matrix_power(a, 3),
    a:matrix_power(3):totable())
  assert_dtype(a:matrix_power(2), "float64")
end)

test("matrix_power supports batches and integer tensors", function()
  local batch = torch.tensor({
    { { 1, 1 }, { 0, 1 } },
    { { 2, 0 }, { 0, 2 } },
  }, { dtype = torch.float64 })
  assert_shape(batch:matrix_power(2), { 2, 2, 2 })
  assert_tensor(batch:matrix_power(2), {
    { { 1, 2 }, { 0, 1 } },
    { { 4, 0 }, { 0, 4 } },
  })
  assert_tensor(batch:matrix_power(0), {
    { { 1, 0 }, { 0, 1 } },
    { { 1, 0 }, { 0, 1 } },
  })
  assert_shape(batch:matrix_power(-1), { 2, 2, 2 })

  local integers = torch.tensor({ { 1, 2 }, { 3, 4 } },
    { dtype = torch.int64 })
  assert_tensor(integers:matrix_power(2), { { 7, 10 }, { 15, 22 } })
  assert_dtype(integers:matrix_power(0), "int64")
  assert_tensor(torch.ones(1, 2, 2, { dtype = torch.int64 }):matrix_power(0),
    { { { 1, 0 }, { 0, 1 } } })
end)

test("matrix_power validates its input", function()
  local a = float_matrix({ { 1, 2 }, { 3, 4 } })
  assert_error(function()
    a:matrix_power(1.5)
  end, "integer exponent")
  assert_error(function()
    a:matrix_power(math.huge)
  end, "integer exponent")
  assert_error(function()
    torch.ones(2, 3):matrix_power(2)
  end, "square matrices")
  assert_error(function()
    torch.ones(2):matrix_power(2)
  end, "square matrices")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):matrix_power(-1)
  end, "floating-point")
  assert_error(function()
    torch.tensor({ { true, false }, { false, true } }):matrix_power(2)
  end, "bool")
end)
