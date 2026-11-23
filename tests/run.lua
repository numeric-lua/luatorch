--
-- luatorch test runner.
--
--   luajit tests/run.lua            # run everything
--   luajit tests/run.lua basic math # run selected files
--
-- Test files are plain scripts that call test("name", function() ... end).
--
local root = arg and arg[0] and arg[0]:match("^(.*)/tests/[^/]+$") or "."
package.path = root .. "/lua/?.lua;" .. root .. "/lua/?/init.lua;" .. package.path

local torch = require("torch")

local passed, failed = 0, 0
local failures = {}
local current_file = "?"
local show_progress = true

function test(name, fn)
  local ok, err = pcall(fn)
  if ok then
    passed = passed + 1
    if show_progress then
      io.write(".")
      io.flush()
    end
  else
    failed = failed + 1
    failures[#failures + 1] = string.format("[%s] %s\n  %s",
      current_file, name, tostring(err):gsub("\n", "\n  "))
    if show_progress then
      io.write("F")
      io.flush()
    end
  end
end

local function describe(value)
  if type(value) == "table" then
    local parts = {}
    for i = 1, #value do
      parts[i] = describe(value[i])
    end
    return "{" .. table.concat(parts, ", ") .. "}"
  end
  return tostring(value)
end

local function compare(actual, expected, tolerance, path)
  if type(expected) == "table" then
    if type(actual) ~= "table" then
      error(string.format("%s: expected a table, got %s", path, type(actual)), 0)
    end
    local count = 0
    for key, value in pairs(expected) do
      count = count + 1
      compare(actual[key], value, tolerance,
        path .. "[" .. tostring(key) .. "]")
    end
    for key in pairs(actual) do
      if expected[key] == nil then
        error(string.format("%s[%s]: unexpected extra value", path, tostring(key)), 0)
      end
    end
    if count == 0 and next(actual) ~= nil then
      error(path .. ": expected an empty table", 0)
    end
    return
  end
  if type(expected) == "number" and type(actual) == "number" then
    if math.abs(actual - expected) > tolerance then
      error(string.format("%s: expected %.9g, got %.9g (tolerance %.3g)",
        path, expected, actual, tolerance), 0)
    end
    return
  end
  if actual ~= expected then
    error(string.format("%s: expected %s, got %s",
      path, tostring(expected), tostring(actual)), 0)
  end
end

function assert_eq(actual, expected, message)
  local ok, err = pcall(compare, actual, expected, 0, "value")
  if not ok then
    error((message or "assert_eq failed") .. "\n  " .. tostring(err), 2)
  end
end

function assert_near(actual, expected, tolerance, message)
  tolerance = tolerance or 1e-5
  local ok, err = pcall(compare, actual, expected, tolerance, "value")
  if not ok then
    error((message or "assert_near failed") .. "\n  " .. tostring(err), 2)
  end
end

function assert_true(value, message)
  if not value then
    error(message or "expected a truthy value", 2)
  end
end

function assert_false(value, message)
  if value then
    error(message or "expected a falsy value", 2)
  end
end

function assert_error(fn, substring)
  local ok, err = pcall(fn)
  if ok then
    error("expected an error" ..
      (substring and (" containing '" .. substring .. "'") or ""), 2)
  end
  local message = tostring(err)
  if substring and not message:find(substring, 1, true) then
    error(string.format("error '%s' does not contain '%s'", message, substring), 2)
  end
end

function assert_tensor(actual, expected, tolerance, message)
  assert_true(torch.is_tensor(actual), message or "expected a Tensor")
  local ok, err = pcall(compare, actual:totable(), expected, tolerance or 1e-5, "tensor")
  if not ok then
    error((message or "assert_tensor failed") .. "\n  " .. tostring(err), 2)
  end
end

function assert_shape(tensor, expected, message)
  assert_eq(tensor:sizes(), expected, message or "unexpected tensor shape")
end

function assert_dtype(tensor, expected, message)
  assert_eq(tensor:dtype(), expected, message or "unexpected tensor dtype")
end

-- deterministic random state for the whole suite
torch.manual_seed(1234)

local requested = {}
for i = 1, #arg do
  requested[#requested + 1] = arg[i]
end

local files = {
  "test_basic.lua",
  "test_math.lua",
  "test_shape.lua",
  "test_reductions.lua",
  "test_tensor_extra.lua",
  "test_blob.lua",
  "test_fuse.lua",
  "test_fusetrain.lua",
  "test_fusereduce.lua",
  "test_linalg.lua",
  "test_linalg_helpers.lua",
  "test_linalg_api.lua",
  "test_random.lua",
  "test_autograd.lua",
  "test_nn.lua",
  "test_nn_extra.lua",
  "test_optim.lua",
  "test_optim_extra.lua",
  "test_data.lua",
  "test_luafun.lua",
  "test_io.lua",
  "test_print.lua",
  "test_einsum.lua",
  "test_float16.lua",
  "test_predicates.lua",
  "test_docs.lua",
  "test_profiler.lua",
  "test_errors.lua",
}

print("luatorch " .. torch.version .. " - " .. root)

local started = os.clock()
for _, file in ipairs(files) do
  local short = file:gsub("^test_", ""):gsub("%.lua$", "")
  local selected = #requested == 0
  for _, name in ipairs(requested) do
    if short == name or file == name then
      selected = true
    end
  end
  if selected then
    current_file = file
    if show_progress then
      io.write("\n" .. string.format("%-22s ", short))
      io.flush()
    end
    local path = root .. "/tests/" .. file
    local chunk, load_error = loadfile(path)
    if not chunk then
      failed = failed + 1
      failures[#failures + 1] = file .. ": " .. tostring(load_error)
    else
      local ok, run_error = pcall(chunk)
      if not ok then
        failed = failed + 1
        failures[#failures + 1] = file .. ": " .. tostring(run_error)
      end
    end
  end
end

if show_progress then
  io.write("\n")
end

local elapsed = os.clock() - started
print(string.format("\n%d passed, %d failed in %.2fs", passed, failed, elapsed))

if #failures > 0 then
  print("\nFAILURES")
  for i, failure in ipairs(failures) do
    print(string.format("\n%d) %s", i, failure))
  end
  os.exit(1)
end
