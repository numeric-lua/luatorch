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

test("solve_triangular solves lower, upper, and unitriangular systems", function()
  local lower = float_matrix({
    { 2, 0, 0 },
    { 1, 3, 0 },
    { 4, -1, 5 },
  })
  local expected = float_matrix({
    { 1, 2 },
    { 3, 4 },
    { 5, 6 },
  })
  local rhs = lower:matmul(expected)
  assert_tensor(lower:solve_triangular(rhs, { upper = false }),
    expected:totable(), 1e-12)
  assert_tensor(torch.linalg.solve_triangular(lower, rhs, { upper = false }),
    expected:totable(), 1e-12)

  local vector = lower:mv(torch.tensor({ 1, 2, 3 }, { dtype = torch.float64 }))
  assert_tensor(lower:solve_triangular(vector, { upper = false }),
    { 1, 2, 3 }, 1e-12)

  local upper = lower:t()
  assert_tensor(upper:solve_triangular(upper:matmul(expected), { upper = true }),
    expected:totable(), 1e-12)

  -- unitriangular ignores the stored diagonal and solves with ones instead.
  local stored = float_matrix({
    { 9, 0, 0 },
    { 1, 8, 0 },
    { 4, -1, 7 },
  })
  local unit = float_matrix({
    { 1, 0, 0 },
    { 1, 1, 0 },
    { 4, -1, 1 },
  })
  assert_tensor(stored:solve_triangular(unit:matmul(expected), {
    upper = false,
    unitriangular = true,
  }), expected:totable(), 1e-12)
end)

test("solve_triangular handles batched and broadcast right-hand sides", function()
  local matrices = {
    { { 3, 0, 0 }, { 1, 4, 0 }, { -1, 2, 5 } },
    { { 2, 0, 0 }, { 3, 6, 0 }, { 1, -2, 7 } },
  }
  local expected = {
    { { 1, 2 }, { 3, 4 }, { 5, 6 } },
    { { -1, 3 }, { 2, 0 }, { 4, -2 } },
  }
  local a = torch.tensor(matrices, { dtype = torch.float64 })
  local x = torch.tensor(expected, { dtype = torch.float64 })
  local rhs = a:matmul(x)
  assert_shape(rhs, { 2, 3, 2 })
  assert_tensor(a:solve_triangular(rhs, { upper = false }),
    expected, 1e-12)

  -- An unbatched RHS broadcasts across the batch dimension.
  local first_rhs = torch.tensor(matrices[1], { dtype = torch.float64 })
    :matmul(x:select(1, 1))
  local broadcast = a:solve_triangular(first_rhs, { upper = false })
  assert_shape(broadcast, { 2, 3, 2 })
  assert_tensor(broadcast:select(1, 1), expected[1], 1e-12)
  assert_tensor(broadcast:select(1, 2),
    torch.tensor(matrices[2], { dtype = torch.float64 })
      :solve_triangular(first_rhs, { upper = false }):totable(), 1e-12)
end)

test("solve_triangular solves right-side systems", function()
  local a = float_matrix({ { 2, 0 }, { 1, 3 } })
  local x = float_matrix({
    { 1, 2 },
    { 3, 4 },
    { 5, 6 },
  })
  local rhs = x:matmul(a)
  assert_shape(rhs, { 3, 2 })
  assert_tensor(a:solve_triangular(rhs, { upper = false, left = false }),
    x:totable(), 1e-12)

  local matrices = {
    { { 2, 0 }, { 1, 3 } },
    { { 4, 0 }, { 2, 6 } },
  }
  local batch_a = torch.tensor(matrices, { dtype = torch.float64 })
  local batch_x = torch.tensor({
    { { 1, 2 }, { 3, 4 } },
    { { 5, 6 }, { 7, 8 } },
  }, { dtype = torch.float64 })
  local batch_rhs = batch_x:matmul(batch_a)
  assert_shape(batch_rhs, { 2, 2, 2 })
  assert_tensor(batch_a:solve_triangular(batch_rhs,
    { upper = false, left = false }), batch_x:totable(), 1e-12)
end)

test("solve_triangular gradients match the implicit function", function()
  local a = torch.tensor({ { 2, 0 }, { 1, 3 } }, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local b = torch.tensor({ { 2 }, { 4 } }, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local weights = float_matrix({ { 3 }, { -1 } })
  local x = a:solve_triangular(b, { upper = false })
  assert_tensor(x, { { 1 }, { 1 } }, 1e-12)
  local loss = (x * weights):sum()
  loss:backward()

  -- For A X = B, with G = A^-T dL/dX: dL/dB = G and
  -- dL/dA = -G X^T, masked to the stored triangle.
  assert_tensor(a:grad(), {
    { -5 / 3, 0 },
    { 1 / 3, 1 / 3 },
  }, 1e-12)
  assert_tensor(b:grad(), { { 5 / 3 }, { -1 / 3 } }, 1e-12)
end)

test("solve_triangular gradients match a full solve on the triangle", function()
  local values = {
    { 4, 0, 0 },
    { 2, 5, 0 },
    { -1, 3, 6 },
  }
  local rhs_values = {
    { 1, 2 },
    { 3, 4 },
    { 5, 6 },
  }
  local weights = float_matrix({
    { 0.5, -1 },
    { 2, 0.25 },
    { -3, 1.5 },
  })

  local triangular_a = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local triangular_b = torch.tensor(rhs_values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local triangular_x = triangular_a:solve_triangular(triangular_b,
    { upper = false })
  local triangular_loss = (triangular_x * weights):sum()
  triangular_loss:backward()

  local full_a = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local full_b = torch.tensor(rhs_values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local full_x = full_a:solve(full_b)
  local full_loss = (full_x * weights):sum()
  full_loss:backward()

  assert_tensor(triangular_x, full_x:totable(), 1e-12)
  assert_tensor(triangular_b:grad(), full_b:grad():totable(), 1e-12)
  for i = 1, 3 do
    for j = 1, 3 do
      if j <= i then
        assert_near(triangular_a:grad():get(i, j),
          full_a:grad():get(i, j), 1e-12)
      else
        assert_near(triangular_a:grad():get(i, j), 0, 1e-15,
          "upper-triangle gradient should stay zero")
      end
    end
  end
end)

test("solve_triangular validates options, shapes, and dtypes", function()
  local a = float_matrix({ { 2, 0 }, { 1, 3 } })
  local b = float_matrix({ { 2 }, { 4 } })
  assert_error(function()
    a:solve_triangular(b)
  end, "requires options")
  assert_error(function()
    a:solve_triangular(b, {})
  end, "upper must be a boolean")
  assert_error(function()
    a:solve_triangular(b, { upper = 1 })
  end, "upper must be a boolean")
  assert_error(function()
    a:solve_triangular(b, { upper = false, left = 1 })
  end, "left must be a boolean")
  assert_error(function()
    a:solve_triangular(b, { upper = false, unitriangular = 1 })
  end, "unitriangular must be a boolean")
  assert_error(function()
    a:solve_triangular(b, { upper = false, transpose = true })
  end, "unknown option")
  assert_error(function()
    a:solve_triangular(b, "upper")
  end, "options must be a table")
  assert_error(function()
    a:solve_triangular({ 2, 4 })
  end, "tensor right-hand side")
  assert_error(function()
    torch.ones(2, 3, { dtype = torch.float64 }):solve_triangular(b,
      { upper = false })
  end, "square matrices")
  assert_error(function()
    a:solve_triangular(torch.ones(3, 1, { dtype = torch.float64 }),
      { upper = false })
  end, "wrong number of rows")
  assert_error(function()
    a:solve_triangular(torch.ones(2, 3, { dtype = torch.float64 }),
      { upper = false, left = false })
  end, "wrong number of columns")
  assert_error(function()
    a:solve_triangular(torch.ones(2, { dtype = torch.float64 }),
      { upper = false, left = false })
  end, "2-D right-hand side")
  assert_error(function()
    a:solve_triangular(torch.ones(3, { dtype = torch.float64 }),
      { upper = false })
  end, "wrong length")
  assert_error(function()
    a:solve_triangular(torch.ones(2, 1, { dtype = torch.float32 }),
      { upper = false })
  end, "match the matrix dtype")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):solve_triangular(
      torch.ones(2, 1, { dtype = torch.int64 }), { upper = false })
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float16 }):solve_triangular(
      torch.ones(2, 1, { dtype = torch.float16 }), { upper = false })
  end, "float32 and float64")
  assert_error(function()
    a:solve_triangular(torch.tensor(1.0, { dtype = torch.float64 }),
      { upper = false })
  end, "vector or matrix")
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

test("lu reconstructs square and rectangular matrices", function()
  -- A zero leading pivot forces a row interchange.
  local swapped = float_matrix({ { 0, 1 }, { 1, 0 } })
  local p, l, u = swapped:lu()
  assert_dtype(p, "float64")
  assert_dtype(l, "float64")
  assert_dtype(u, "float64")
  assert_tensor(p, { { 0, 1 }, { 1, 0 } })
  assert_tensor(l, { { 1, 0 }, { 0, 1 } })
  assert_tensor(u, { { 1, 0 }, { 0, 1 } })
  assert_tensor(p:matmul(l):matmul(u), swapped:totable(), 1e-12)

  -- No interchange, with nontrivial L and U entries.
  local square = float_matrix({
    { 4, 3, 2 },
    { 1, 5, 7 },
    { 2, 2, 6 },
  })
  local sp, sl, su = torch.linalg.lu(square)
  assert_tensor(sp, { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } })
  assert_near(sl:totable(), {
    { 1, 0, 0 },
    { 0.25, 1, 0 },
    { 0.5, 0.11764705882352941, 1 },
  }, 1e-12)
  assert_near(su:totable(), {
    { 4, 3, 2 },
    { 0, 4.25, 6.5 },
    { 0, 0, 4.235294117647059 },
  }, 1e-12)
  assert_tensor(sp:matmul(sl):matmul(su), square:totable(), 1e-12)

  -- Rectangular inputs return P (m x m), L (m x min(m, n)), U (min(m, n) x n).
  local wide = float_matrix({
    { 1, 2, 3 },
    { 4, 5, 6 },
  })
  local wp, wl, wu = wide:lu()
  assert_shape(wp, { 2, 2 })
  assert_shape(wl, { 2, 2 })
  assert_shape(wu, { 2, 3 })
  assert_tensor(wp, { { 0, 1 }, { 1, 0 } })
  assert_near(wl:totable(), { { 1, 0 }, { 0.25, 1 } }, 1e-12)
  assert_near(wu:totable(), { { 4, 5, 6 }, { 0, 0.75, 1.5 } }, 1e-12)
  assert_tensor(wp:matmul(wl):matmul(wu), wide:totable(), 1e-12)

  local tall = float_matrix({
    { 1, 2 },
    { 3, 4 },
    { 5, 6 },
  })
  local tp, tl, tu = tall:lu()
  assert_shape(tp, { 3, 3 })
  assert_shape(tl, { 3, 2 })
  assert_shape(tu, { 2, 2 })
  assert_tensor(tp:matmul(tl):matmul(tu), tall:totable(), 1e-12)
  for i = 1, 2 do
    assert_near(tl:get(i, i), 1, 1e-12, "L diagonal must be one")
  end
  for i = 1, 3 do
    for j = i + 1, 2 do
      assert_near(tl:get(i, j), 0, 1e-12, "L must be lower triangular")
    end
  end
  for i = 2, 2 do
    for j = 1, i - 1 do
      assert_near(tu:get(i, j), 0, 1e-12, "U must be upper triangular")
    end
  end
end)

test("lu handles batches and empty matrices", function()
  local batch = torch.tensor({
    { { 1, 2, 3 }, { 4, 5, 6 } },
    { { 0, 1, 2 }, { 3, 4, 5 } },
  }, { dtype = torch.float64 })
  local p, l, u = batch:lu()
  assert_shape(p, { 2, 2, 2 })
  assert_shape(l, { 2, 2, 2 })
  assert_shape(u, { 2, 2, 3 })
  assert_tensor(p:bmm(l):bmm(u), batch:totable(), 1e-12)
  assert_tensor(p:select(1, 1), { { 0, 1 }, { 1, 0 } })
  assert_tensor(p:select(1, 2), { { 0, 1 }, { 1, 0 } })

  local empty_p, empty_l, empty_u =
    torch.zeros(0, 0, { dtype = torch.float64 }):lu()
  assert_shape(empty_p, { 0, 0 })
  assert_shape(empty_l, { 0, 0 })
  assert_shape(empty_u, { 0, 0 })
end)

test("lu gradients flow through L and U", function()
  local values = { { 4, 1 }, { 2, 3 } }
  local a = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local p, l, u = a:lu()
  assert_true(not p:requires_grad(), "P is a permutation matrix")
  assert_true(l:requires_grad())
  assert_true(u:requires_grad())

  local weight_l = float_matrix({ { 1, 2 }, { 3, 5 } })
  local weight_u = float_matrix({ { -1, 0.5 }, { 2, 4 } })
  local loss = (l * weight_l):sum() + (u * weight_u):sum()
  loss:backward()

  -- The pivots do not move for these small perturbations, so a central
  -- difference checks the ATen LU backward for both factors.
  local eps = 1e-6
  local function perturbed(i, j, delta)
    local copy = torch.tensor(values, { dtype = torch.float64 })
    copy:set(i, j, values[i][j] + delta)
    local _, pl, pu = copy:lu()
    return ((pl * weight_l):sum() + (pu * weight_u):sum()):item()
  end
  for i = 1, 2 do
    for j = 1, 2 do
      local numeric = (perturbed(i, j, eps) - perturbed(i, j, -eps)) /
        (2 * eps)
      assert_near(a:grad():get(i, j), numeric, 1e-5,
        string.format("d loss / d A[%d][%d]", i, j))
    end
  end
end)

test("lu validates dimensions and dtypes", function()
  assert_error(function()
    torch.ones(3):lu()
  end, "matrix or a batch")
  assert_error(function()
    torch.tensor(1.0):lu()
  end, "matrix or a batch")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):lu()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float16 }):lu()
  end, "float32 and float64")
  assert_error(function()
    torch.tensor({ { true, false }, { false, true } }):lu()
  end, "float32 and float64")

  local p, l, u = torch.ones(2, 3):lu()
  assert_dtype(p, "float32")
  assert_dtype(l, "float32")
  assert_dtype(u, "float32")
end)

-- Rebuild P, L, U from the packed LU and 1-based pivots returned by
-- lu_factor. The row swaps are applied in reverse order, matching
-- torch.lu_unpack's convention for A = P L U.
local function unpack_lu(lu, pivots)
  local m, n = lu:size(1), lu:size(2)
  local k = math.min(m, n)
  local dtype = lu:dtype()
  local l = torch.zeros(m, k, { dtype = dtype })
  local u = torch.zeros(k, n, { dtype = dtype })
  for i = 1, k do
    l:set(i, i, 1)
  end
  for i = 1, m do
    for j = 1, n do
      local value = lu:get(i, j)
      if i > j and j <= k then
        l:set(i, j, value)
      elseif i <= j and i <= k then
        u:set(i, j, value)
      end
    end
  end
  local p = torch.zeros(m, m, { dtype = dtype })
  for i = 1, m do
    p:set(i, i, 1)
  end
  for i = k, 1, -1 do
    local j = pivots:get(i)
    if j ~= i then
      for column = 1, m do
        local left = p:get(i, column)
        p:set(i, column, p:get(j, column))
        p:set(j, column, left)
      end
    end
  end
  return p, l, u
end

test("lu_factor returns a compact pivoted factorization", function()
  local a = float_matrix({ { 4, 3 }, { 2, 1 } })
  local lu, pivots = a:lu_factor()
  assert_dtype(lu, "float64")
  assert_dtype(pivots, "int32")
  assert_tensor(lu, { { 4, 3 }, { 0.5, -0.5 } }, 1e-12)
  assert_eq(pivots:totable(), { 1, 2 })

  local p, l, u = unpack_lu(lu, pivots)
  assert_tensor(p, { { 1, 0 }, { 0, 1 } })
  assert_tensor(l, { { 1, 0 }, { 0.5, 1 } }, 1e-12)
  assert_tensor(u, { { 4, 3 }, { 0, -0.5 } }, 1e-12)
  assert_tensor(p:matmul(l):matmul(u), a:totable(), 1e-12)

  -- A zero leading pivot forces a row interchange; pivots are 1-based.
  local swapped = float_matrix({ { 0, 1 }, { 1, 0 } })
  local swapped_lu, swapped_pivots = swapped:lu_factor()
  assert_tensor(swapped_lu, { { 1, 0 }, { 0, 1 } })
  assert_eq(swapped_pivots:totable(), { 2, 2 })
  local sp, sl, su = unpack_lu(swapped_lu, swapped_pivots)
  assert_tensor(sp, { { 0, 1 }, { 1, 0 } })
  assert_tensor(sp:matmul(sl):matmul(su), swapped:totable(), 1e-12)

  -- The packed layout stores U on and above the diagonal and L below it.
  local _, l_factor, u_factor = a:lu()
  for i = 1, 2 do
    for j = 1, 2 do
      local expected = i > j and l_factor:get(i, j) or u_factor:get(i, j)
      assert_near(lu:get(i, j), expected, 1e-12,
        string.format("packed LU[%d][%d]", i, j))
    end
  end

  local float32_lu, float32_pivots =
    torch.tensor({ { 4, 3 }, { 2, 1 } }):lu_factor()
  assert_dtype(float32_lu, "float32")
  assert_dtype(float32_pivots, "int32")
end)

test("lu_factor handles rectangular, batched, and empty matrices", function()
  local tall = float_matrix({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local tall_lu, tall_pivots = tall:lu_factor()
  assert_shape(tall_lu, { 3, 2 })
  assert_shape(tall_pivots, { 2 })
  assert_tensor(tall_lu, {
    { 5, 6 },
    { 0.2, 0.7999999999999998 },
    { 0.6000000000000001, 0.49999999999999944 },
  }, 1e-12)
  assert_eq(tall_pivots:totable(), { 3, 3 })
  local tp, tl, tu = unpack_lu(tall_lu, tall_pivots)
  assert_tensor(tp:matmul(tl):matmul(tu), tall:totable(), 1e-12)

  local wide = float_matrix({ { 1, 2, 3 }, { 4, 5, 6 } })
  local wide_lu, wide_pivots = wide:lu_factor()
  assert_shape(wide_lu, { 2, 3 })
  assert_shape(wide_pivots, { 2 })
  assert_tensor(wide_lu, { { 4, 5, 6 }, { 0.25, 0.75, 1.5 } }, 1e-12)
  assert_eq(wide_pivots:totable(), { 2, 2 })
  local wp, wl, wu = unpack_lu(wide_lu, wide_pivots)
  assert_tensor(wp:matmul(wl):matmul(wu), wide:totable(), 1e-12)

  local batch = torch.tensor({
    { { 4, 1, 2 }, { 3, 5, 1 }, { 1, 2, 6 } },
    { { 0, 2, 1 }, { 1, 0, 3 }, { 4, 1, 0 } },
  }, { dtype = torch.float64 })
  local batch_lu, batch_pivots = batch:lu_factor()
  assert_shape(batch_lu, { 2, 3, 3 })
  assert_shape(batch_pivots, { 2, 3 })
  assert_eq(batch_pivots:totable(), {
    { 1, 2, 3 },
    { 3, 3, 3 },
  })
  for i = 1, 2 do
    local bp, bl, bu = unpack_lu(batch_lu:select(1, i),
      batch_pivots:select(1, i))
    assert_tensor(bp:matmul(bl):matmul(bu),
      batch:select(1, i):totable(), 1e-12)
  end

  local empty_lu, empty_pivots =
    torch.zeros(0, 0, { dtype = torch.float64 }):lu_factor()
  assert_shape(empty_lu, { 0, 0 })
  assert_shape(empty_pivots, { 0 })
  local batch_empty_lu, batch_empty_pivots =
    torch.zeros(2, 0, 0, { dtype = torch.float64 }):lu_factor()
  assert_shape(batch_empty_lu, { 2, 0, 0 })
  assert_shape(batch_empty_pivots, { 2, 0 })
end)

test("lu_solve solves systems from a compact factorization", function()
  local a = float_matrix({ { 4, 3 }, { 2, 1 } })
  local lu, pivots = a:lu_factor()
  local single = float_matrix({ { 1 }, { 2 } })
  assert_tensor(lu:lu_solve(pivots, single), { { 2.5 }, { -3 } }, 1e-12)
  assert_tensor(torch.linalg.lu_solve(lu, pivots, single),
    { { 2.5 }, { -3 } }, 1e-12)

  local multiple = float_matrix({ { 1, 2, 3 }, { 4, 5, 6 } })
  assert_tensor(lu:lu_solve(pivots, multiple), {
    { 5.5, 6.5, 7.5 },
    { -7, -8, -9 },
  }, 1e-12)
  assert_tensor(lu:lu_solve(pivots, multiple),
    a:solve(multiple):totable(), 1e-12)

  -- X A = B, with three right-hand-side rows but the same 2x2 factor.
  local right = float_matrix({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  assert_tensor(lu:lu_solve(pivots, right, { left = false }), {
    { 1.5, -2.5 },
    { 2.5, -3.5 },
    { 3.5, -4.5 },
  }, 1e-12)

  -- adjoint solves A^T X = B for real tensors.
  assert_tensor(lu:lu_solve(pivots, single, { adjoint = true }),
    { { 1.5 }, { -2.5 } }, 1e-12)
  assert_tensor(lu:lu_solve(pivots, single, { adjoint = true }),
    a:t():solve(single):totable(), 1e-12)

  local batch = torch.tensor({
    { { 4, 1, 2 }, { 3, 5, 1 }, { 1, 2, 6 } },
    { { 0, 2, 1 }, { 1, 0, 3 }, { 4, 1, 0 } },
  }, { dtype = torch.float64 })
  local batch_lu, batch_pivots = batch:lu_factor()
  local rhs = torch.tensor({
    { { 1, 2 }, { 3, 4 }, { 5, 6 } },
    { { -1, 0 }, { 2, 1 }, { 0, 3 } },
  }, { dtype = torch.float64 })
  assert_tensor(batch_lu:lu_solve(batch_pivots, rhs),
    batch:solve(rhs):totable(), 1e-12)

  -- A right-hand side without batch dimensions broadcasts over the factor.
  local shared = float_matrix({
    { 1, 0, 0 },
    { 0, 1, 0 },
    { 0, 0, 1 },
  })
  assert_tensor(batch_lu:lu_solve(batch_pivots, shared),
    batch:solve(shared):totable(), 1e-12)

  local right_batch = torch.tensor({
    { { 1, 2, 3 }, { 4, 5, 6 } },
    { { 0, 1, 2 }, { 3, 4, 5 } },
  }, { dtype = torch.float64 })
  local expected_right = right_batch:matmul(batch:inverse())
  assert_tensor(batch_lu:lu_solve(batch_pivots, right_batch,
    { left = false }), expected_right:totable(), 1e-12)
end)

test("lu_factor and lu_solve gradients match the implicit derivative", function()
  local values = { { 4, 1 }, { 2, 3 } }
  local a = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local b = torch.tensor({ { 1, 2 }, { 3, 4 } }, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local lu, pivots = a:lu_factor()
  assert_true(lu:requires_grad(), "LU carries gradients")
  assert_true(not pivots:requires_grad(), "pivots are integer indices")

  local reference_a = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local reference_b = torch.tensor({ { 1, 2 }, { 3, 4 } }, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local weight = float_matrix({ { 1, 2 }, { 3, 5 } })
  local solution = lu:lu_solve(pivots, b)
  local loss = (solution * weight):sum()
  loss:backward()
  local reference = reference_a:solve(reference_b)
  local reference_loss = (reference * weight):sum()
  reference_loss:backward()
  assert_near(a:grad():totable(), reference_a:grad():totable(), 1e-10)
  assert_near(b:grad():totable(), reference_b:grad():totable(), 1e-10)

  -- The adjoint solve differentiates like solving with A^T.
  local adjoint_a = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local adjoint_b = torch.tensor({ { 1, 2 }, { 3, 4 } }, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local transposed_a = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local transposed_b = torch.tensor({ { 1, 2 }, { 3, 4 } }, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local adjoint_lu, adjoint_pivots = adjoint_a:lu_factor()
  local adjoint_solution = adjoint_lu:lu_solve(adjoint_pivots, adjoint_b,
    { adjoint = true })
  local adjoint_loss = (adjoint_solution * weight):sum()
  adjoint_loss:backward()
  local transposed_solution = transposed_a:t():solve(transposed_b)
  local transposed_loss = (transposed_solution * weight):sum()
  transposed_loss:backward()
  assert_near(adjoint_a:grad():totable(), transposed_a:grad():totable(),
    1e-10)
  assert_near(adjoint_b:grad():totable(), transposed_b:grad():totable(),
    1e-10)

  -- Central differences check the factor gradient directly. The pivots do
  -- not move for these perturbations, so the factorization is smooth here.
  local factor_a = torch.tensor(values, {
    dtype = torch.float64,
    requires_grad = true,
  })
  local factor_lu = factor_a:lu_factor()
  local factor_loss = (factor_lu * weight):sum()
  factor_loss:backward()

  local eps = 1e-6
  local function packed_loss(perturbed)
    local tensor = torch.tensor(perturbed, { dtype = torch.float64 })
    local packed = tensor:lu_factor()
    return (packed * weight):sum():item()
  end
  for i = 1, 2 do
    for j = 1, 2 do
      local plus = { { 4, 1 }, { 2, 3 } }
      local minus = { { 4, 1 }, { 2, 3 } }
      plus[i][j] = plus[i][j] + eps
      minus[i][j] = minus[i][j] - eps
      local numeric = (packed_loss(plus) - packed_loss(minus)) / (2 * eps)
      assert_near(factor_a:grad():get(i, j), numeric, 1e-5,
        string.format("d packed LU loss / d A[%d][%d]", i, j))
    end
  end
end)

test("lu_factor and lu_solve validate shapes, dtypes, and options", function()
  local a = float_matrix({ { 4, 3 }, { 2, 1 } })
  local lu, pivots = a:lu_factor()

  assert_error(function()
    torch.ones(3):lu_factor()
  end, "matrix or a batch")
  assert_error(function()
    torch.tensor(1.0):lu_factor()
  end, "matrix or a batch")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):lu_factor()
  end, "float32 and float64")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.float16 }):lu_factor()
  end, "float32 and float64")
  assert_error(function()
    torch.tensor({ { true, false }, { false, true } }):lu_factor()
  end, "float32 and float64")
  assert_error(function()
    float_matrix({ { 1, 2 }, { 2, 4 } }):lu_factor()
  end, "is zero")

  assert_error(function()
    lu:lu_solve(2, float_matrix({ { 1 }, { 2 } }))
  end, "int32 pivots tensor")
  assert_error(function()
    lu:lu_solve(pivots, { 1, 2 })
  end, "tensor right-hand side")
  assert_error(function()
    lu:lu_solve(pivots, float_matrix({ { 1 }, { 2 } }), "left")
  end, "options must be a table")
  assert_error(function()
    lu:lu_solve(pivots, float_matrix({ { 1 }, { 2 } }), { transpose = true })
  end, "unknown option")
  assert_error(function()
    lu:lu_solve(pivots, float_matrix({ { 1 }, { 2 } }), { left = 1 })
  end, "left must be a boolean")
  assert_error(function()
    lu:lu_solve(pivots, float_matrix({ { 1 }, { 2 } }), { adjoint = "yes" })
  end, "adjoint must be a boolean")

  assert_error(function()
    torch.ones(2, 3, { dtype = torch.float64 }):lu_solve(pivots,
      float_matrix({ { 1 }, { 2 } }))
  end, "square matrices")
  assert_error(function()
    torch.ones(3):lu_solve(pivots, float_matrix({ { 1 }, { 2 } }))
  end, "square matrices")
  assert_error(function()
    torch.ones(2, 2, { dtype = torch.int64 }):lu_solve(pivots,
      float_matrix({ { 1 }, { 2 } }))
  end, "float32 and float64")
  assert_error(function()
    lu:lu_solve(pivots:float(), float_matrix({ { 1 }, { 2 } }))
  end, "int32 dtype")
  assert_error(function()
    lu:lu_solve(torch.zeros(2, { dtype = torch.int64 }),
      float_matrix({ { 1 }, { 2 } }))
  end, "int32 dtype")
  assert_error(function()
    lu:lu_solve(pivots,
      torch.ones(2, 1, { dtype = torch.float32 }))
  end, "match the matrix dtype")
  assert_error(function()
    lu:lu_solve(torch.zeros(2, 2, { dtype = torch.int32 }),
      float_matrix({ { 1 }, { 2 } }))
  end, "one dimension less")
  assert_error(function()
    lu:lu_solve(torch.zeros(1, { dtype = torch.int32 }),
      float_matrix({ { 1 }, { 2 } }))
  end, "wrong length")
  assert_error(function()
    lu:lu_solve(pivots, float_matrix({ 1, 2 }))
  end, "at least 2 dimensions")
  assert_error(function()
    lu:lu_solve(pivots, float_matrix({ { 1 }, { 2 }, { 3 } }))
  end, "wrong number of rows")
  assert_error(function()
    lu:lu_solve(pivots, float_matrix({ { 1, 2, 3 } }), { left = false })
  end, "wrong number of columns")

  local batch_lu, batch_pivots = torch.tensor({
    { { 3, 1 }, { 1, 2 } },
    { { 4, 2 }, { 1, 3 } },
  }, { dtype = torch.float64 }):lu_factor()
  assert_shape(batch_lu, { 2, 2, 2 })
  assert_shape(batch_pivots, { 2, 2 })
  assert_error(function()
    batch_lu:lu_solve(torch.tensor({ { 1, 2 }, { 1, 2 }, { 1, 2 } },
      { dtype = torch.int32 }), float_matrix({ { 1 }, { 2 } }))
  end, "batch dimensions")
  assert_error(function()
    lu:lu_solve(batch_pivots, float_matrix({ { 1 }, { 2 } }))
  end, "one dimension less")
  assert_error(function()
    batch_lu:lu_solve(pivots, float_matrix({ { 1 }, { 2 } }))
  end, "one dimension less")
end)
