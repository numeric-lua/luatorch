--
-- luatorch micro-benchmarks.
--
--   luajit bench/bench.lua
--
-- Numbers are wall-clock-ish (os.clock, single-threaded) and meant for
-- relative comparisons on the same machine, not as authoritative results.
--
local root = (arg and arg[0] and arg[0]:match("^(.*)/bench/[^/]+$")) or "."
package.path = root .. "/lua/?.lua;" .. root .. "/lua/?/init.lua;" .. package.path

local torch = require("torch")
local nn = require("torch.nn")
local optim = require("torch.optim")
local fun = require("fun")

local results = {}

local function median(times)
  table.sort(times)
  return times[math.floor(#times / 2) + 1]
end

local function bench(name, iterations, fn, note)
  fn() -- warmup
  collectgarbage("collect")
  local times = {}
  for i = 1, iterations do
    local started = os.clock()
    fn()
    times[i] = os.clock() - started
  end
  local seconds = median(times)
  results[#results + 1] = { name = name, seconds = seconds, note = note }
  return seconds
end

local function format_row(row)
  local milliseconds = row.seconds * 1000
  local per_iteration = row.note or ""
  return string.format("%-34s %12.3f ms  %s", row.name, milliseconds, per_iteration)
end

print(string.format("luatorch %s benchmarks (libtorch %s)", torch.VERSION or "0.7.0", torch.version))
print(string.rep("-", 78))

torch.manual_seed(42)

-- creation --------------------------------------------------------------
bench("zeros(1000, 1000)", 20, function()
  torch.zeros(1000, 1000)
end, "8 MB")

bench("randn(1000, 1000)", 20, function()
  torch.randn(1000, 1000)
end, "8 MB")

-- elementwise -----------------------------------------------------------
local a = torch.randn(1000000)
local b = torch.randn(1000000)

local add_seconds = bench("add: 1M float32 tensors", 30, function()
  local c = a + b
end, "12 MB traffic")
print(string.format("%-34s %12.1f GB/s", "  effective bandwidth",
  (3 * 4 * 1000000) / add_seconds / 1e9))

bench("add scalar: 1M float32", 30, function()
  local c = a + 1.5
end)

bench("sum: 1M float32", 30, function()
  local s = a:sum()
end)

bench("relu: 1M float32", 30, function()
  local c = a:relu()
end)

-- linear algebra --------------------------------------------------------
for _, size in ipairs({ 128, 256, 512 }) do
  local x = torch.randn(size, size)
  local y = torch.randn(size, size)
  local seconds = bench(string.format("matmul: %dx%d float32", size, size), 10, function()
    local z = x:mm(y)
  end)
  local gflops = (2 * size ^ 3) / seconds / 1e9
  print(string.format("%-34s %12.1f GFLOP/s", "  throughput", gflops))
end

-- sorting and linear algebra ------------------------------------------
local million = torch.randn(1000000)
bench("sort 1M float32", 5, function()
  local values, indices = million:sort(1)
end)
bench("topk 1M, k=10", 10, function()
  local values, indices = million:topk(10)
end)

local square = torch.randn(128, 128)
bench("svd 128x128 float32", 5, function()
  local u, s, vh = square:svd(false)
end)

bench("pinv 128x128 float32", 5, function()
  local inverse = square:pinv()
end)

local pinv_tall = torch.randn(256, 128)
bench("pinv 256x128 float32", 5, function()
  local inverse = pinv_tall:pinv()
end, "tall")

local pinv_grad = torch.randn(64, 64, { requires_grad = true })
bench("pinv fwd+bwd 64x64 float32", 10, function()
  local inverse = pinv_grad:pinv()
  inverse:sum():backward()
  pinv_grad:zero_grad()
end, "includes zero_grad")

-- recent ops: einsum, float16, predicates, slogdet -----------------------
local einsum_a = torch.randn(128, 128)
local einsum_b = torch.randn(128, 128)
bench("einsum ij,jk->ik 128x128", 20, function()
  local c = torch.einsum("ij,jk->ik", einsum_a, einsum_b)
end)

local batched_a = torch.randn(8, 64, 64)
local batched_b = torch.randn(8, 64, 64)
bench("einsum batched 8x64x64", 10, function()
  local c = torch.einsum("...ij,...jk->...ik", batched_a, batched_b)
end)

local grad_a = torch.randn(64, 64, { requires_grad = true })
local grad_b = torch.randn(64, 64, { requires_grad = true })
bench("einsum fwd+bwd 64x64", 10, function()
  local out = torch.einsum("ij,jk->ik", grad_a, grad_b)
  out:sum():backward()
  grad_a:zero_grad()
  grad_b:zero_grad()
end, "includes zero_grad")

local half_a = torch.randn(256, 256, { dtype = torch.float16 })
local half_b = torch.randn(256, 256, { dtype = torch.float16 })
bench("matmul 256x256 float16", 10, function()
  local c = half_a:mm(half_b)
end)

local half_million = torch.randn(1000000, { dtype = torch.float16 })
bench("add 1M float16", 20, function()
  local c = half_million + half_million
end)

bench("isfinite 1M float32", 20, function()
  local mask = torch.isfinite(million)
end)

bench("slogdet 128x128 float32", 10, function()
  local sign, logabsdet = square:slogdet()
end)

local tall = torch.randn(256, 128)
local tall_rhs = torch.randn(256)
bench("lstsq 256x128 float32", 5, function()
  local solution, residuals, rank, singular_values =
    tall:lstsq(tall_rhs, { driver = "gelsd" })
end, "gelsd")

local lstsq_a = torch.randn(128, 64, { requires_grad = true })
local lstsq_b = torch.randn(128, { requires_grad = true })
bench("lstsq fwd+bwd 128x64 float32", 5, function()
  local solution = lstsq_a:lstsq(lstsq_b, { driver = "gelsd" })
  solution:sum():backward()
  lstsq_a:zero_grad()
  lstsq_b:zero_grad()
end, "gelsd, includes zero_grad")

local function lower_triangular(size)
  local values = torch.randn(size, size):totable()
  for i = 1, size do
    for j = i + 1, size do
      values[i][j] = 0
    end
    values[i][i] = values[i][i] + size
  end
  return torch.tensor(values)
end

local triangular_a = lower_triangular(128)
local triangular_b = torch.randn(128, 128)
bench("solve_triangular 128x128 f32", 10, function()
  local solution = triangular_a:solve_triangular(triangular_b, { upper = false })
end, "lower")

local triangular_grad_a = lower_triangular(64)
triangular_grad_a:requires_grad_(true)
local triangular_grad_b = torch.randn(64, 64, { requires_grad = true })
bench("solve_triangular fwd+bwd 64x64", 10, function()
  local solution = triangular_grad_a:solve_triangular(triangular_grad_b,
    { upper = false })
  solution:sum():backward()
  triangular_grad_a:zero_grad()
  triangular_grad_b:zero_grad()
end, "lower, includes zero_grad")

local lu_square = torch.randn(128, 128)
bench("lu 128x128 float32", 10, function()
  local p, l, u = lu_square:lu()
end, "pivoted")

local lu_grad = torch.randn(64, 64, { requires_grad = true })
bench("lu fwd+bwd 64x64 float32", 10, function()
  local p, l, u = lu_grad:lu()
  local loss = l:sum() + u:sum()
  loss:backward()
  lu_grad:zero_grad()
end, "pivoted, includes zero_grad")

local factor_square = torch.randn(128, 128)
bench("lu_factor 128x128 f32", 10, function()
  local factorization, pivots = factor_square:lu_factor()
end, "packed, pivoted")

local factor_lu, factor_pivots = factor_square:lu_factor()
local factor_rhs = torch.randn(128, 32)
bench("lu_solve 128x128x32 f32", 10, function()
  local solution = factor_lu:lu_solve(factor_pivots, factor_rhs)
end, "32 RHS")

local factor_grad = torch.randn(64, 64, { requires_grad = true })
local factor_grad_rhs = torch.randn(64, 16, { requires_grad = true })
bench("lu_factor+solve fwd+bwd 64x64", 10, function()
  local factorization, pivots = factor_grad:lu_factor()
  local solution = factorization:lu_solve(pivots, factor_grad_rhs)
  solution:sum():backward()
  factor_grad:zero_grad()
  factor_grad_rhs:zero_grad()
end, "16 RHS, includes zero_grad")

local power_base = torch.randn(128, 128)
bench("matrix_power 128x128 p=8", 10, function()
  local powered = power_base:matrix_power(8)
end)

local power_grad = torch.randn(64, 64, { requires_grad = true })
bench("matrix_power fwd+bwd 64x64 p=8", 10, function()
  local powered = power_grad:matrix_power(8)
  powered:sum():backward()
  power_grad:zero_grad()
end, "p=8, includes zero_grad")

local trapezoid_values = torch.randn(4096)
bench("trapezoid 4096 float32", 50, function()
  local integral = torch.trapezoid(trapezoid_values)
end)

local trapezoid_grad = torch.randn(1024, { requires_grad = true })
bench("trapezoid fwd+bwd 1024 float32", 50, function()
  local integral = torch.trapezoid(trapezoid_grad, { dx = 0.5 })
  integral:backward()
  trapezoid_grad:zero_grad()
end, "includes zero_grad")

-- per-call overhead -----------------------------------------------------
local tiny = torch.randn(16)
bench("small tensor add (16 elements)", 20000, function()
  local c = tiny + tiny
end, "per call")

bench("scalar item read", 20000, function()
  local value = tiny:get(1)
end, "per call")

-- Lua-side iteration ----------------------------------------------------
local values = torch.randn(200000):totable()

local plain_seconds = bench("plain Lua loop over 200k", 10, function()
  local total = 0
  for i = 1, #values do
    total = total + values[i]
  end
end)

local fun_seconds = bench("luafun map+reduce over 200k", 10, function()
  local total = fun.iter(values)
    :map(function(x) return x * 2 end)
    :reduce(function(acc, x) return acc + x end, 0)
end)

print(string.format("%-34s %12.2fx", "  luafun vs plain loop", fun_seconds / plain_seconds))

local tensor = torch.randn(200000)

bench("tensor:iter sum 200k", 10, function()
  local total = 0
  for value in tensor:iter() do
    total = total + value
  end
end)

bench("tensor:totable 200k", 10, function()
  local t = tensor:totable()
end, "allocates a Lua table")

-- serialization ---------------------------------------------------------
local path = os.tmpname() .. ".pt"
local saved = torch.randn(1000, 100)
bench("torch.save 100k floats", 10, function()
  saved:save(path)
end, "PyTorch zip format")
bench("torch.load 100k floats", 10, function()
  torch.load(path)
end)
os.remove(path)

-- neural networks -------------------------------------------------------
local mlp = nn.Sequential(
  nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
local batch = torch.randn(128, 784)
local labels = torch.LongTensor(128):fill_(1)
local criterion = nn.CrossEntropyLoss()
local adam = optim.Adam(mlp:parameters(), { lr = 1e-3 })

bench("MLP forward 128x784", 50, function()
  local output = mlp(batch)
end)

bench("MLP forward+backward 128x784", 30, function()
  local output = mlp(batch)
  local loss = criterion(output, labels)
  mlp:zero_grad()
  loss:backward()
end)

bench("MLP Adam step 128x784", 30, function()
  local output = mlp(batch)
  local loss = criterion(output, labels)
  adam:zero_grad()
  loss:backward()
  adam:step()
end)

local convnet = nn.Sequential(
  nn.Conv2d(3, 16, 3, { padding = 1 }),
  nn.ReLU(),
  nn.MaxPool2d(2),
  nn.Conv2d(16, 32, 3, { padding = 1 }),
  nn.ReLU(),
  nn.MaxPool2d(2))
local images = torch.randn(8, 3, 32, 32)

bench("ConvNet forward 8x3x32x32", 20, function()
  local output = convnet(images)
end)

print(string.rep("-", 78))
print(string.format("%-34s %12s", "benchmark", "median"))

for _, row in ipairs(results) do
  print(format_row(row))
end
