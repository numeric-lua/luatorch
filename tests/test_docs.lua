local torch = require("torch")

local source = debug.getinfo(1, "S").source
local root = source:match("^@?(.*)/tests/[^/]+$") or "."

local function read(path)
  local file = assert(io.open(root .. "/" .. path, "rb"),
    "cannot read " .. path)
  local contents = file:read("*a")
  file:close()
  return contents
end

local function exists(path)
  local file = io.open(root .. "/" .. path, "rb")
  if file then file:close() end
  return file ~= nil
end

test("release version is consistent across docs and packaging", function()
  local version = torch.VERSION
  assert_true(read("docs/api.md"):find(
    "# luatorch API reference (" .. version .. ")", 1, true) ~= nil,
    "docs/api.md heading does not name " .. version)
  assert_true(exists("luatorch-" .. version .. "-1.rockspec"),
    "missing rockspec for " .. version)
  assert_true(read("CHANGELOG.md"):find(
    "## [" .. version .. "]", 1, true) ~= nil,
    "CHANGELOG.md has no " .. version .. " section")
  assert_true(read("RELEASE_NOTES.md"):find(
    "# luatorch " .. version, 1, true) ~= nil,
    "RELEASE_NOTES.md does not name " .. version)
  assert_true(read("README.md"):find(
    "**" .. version .. " alpha**", 1, true) ~= nil,
    "README.md status does not name " .. version)
end)

test("documented public functions exist", function()
  local api = read("docs/api.md")
  for _, name in ipairs({ "einsum", "isnan", "isinf", "isfinite", "from_blob",
      "trapezoid", "lu_unpack", "diagonal", "vander", "cross" }) do
    assert_true(api:find("torch." .. name, 1, true) ~= nil,
      "docs/api.md does not mention torch." .. name)
    assert_eq(type(torch[name]), "function", "torch." .. name .. " is missing")
  end
end)

test("rockspec modules resolve to source files", function()
  local rockspec = read("luatorch-" .. torch.VERSION .. "-1.rockspec")
  local count = 0
  for module, path in rockspec:gmatch('%["([%w%._]+)"%]%s*=%s*"([^"]+)"') do
    assert_true(exists(path),
      "rockspec module " .. module .. " points at missing " .. path)
    count = count + 1
  end
  assert_true(count >= 10, "rockspec module list looks truncated")
end)

test("documented linalg aliases exist", function()
  local api = read("docs/api.md")
  for _, name in ipairs({ "inv", "inv_ex", "tensorinv", "tensorsolve", "det",
      "slogdet", "solve", "solve_ex",
      "solve_triangular", "lstsq", "cholesky", "cholesky_ex", "eigh", "eigvalsh",
      "svd", "qr",
      "householder_product", "lu", "lu_factor", "lu_factor_ex", "lu_solve", "pinv",
      "ldl_factor", "ldl_factor_ex", "ldl_solve",
      "matrix_exp", "matrix_power",
      "svdvals", "matrix_norm", "norm", "cond", "matrix_rank", "vector_norm",
      "diagonal",
      "vander", "cross", "vecdot", "multi_dot" }) do
    assert_true(api:find("torch.linalg." .. name, 1, true) ~= nil,
      "docs/api.md does not mention torch.linalg." .. name)
    assert_eq(type(torch.linalg[name]), "function",
      "torch.linalg." .. name .. " is missing")
  end
end)

test("predicate documentation matches integer and bool behavior", function()
  local api = read("docs/api.md")
  assert_true(api:find("integer and bool tensors produce all-true", 1, true) ~= nil,
    "docs/api.md does not state that integer/bool predicates are all-true")
  assert_true(api:find("all-false", 1, true) == nil,
    "docs/api.md still claims all-false integer/bool masks")
end)
