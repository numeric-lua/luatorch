--
-- torch.nn.linear, activations, and containers.
--
local torch = require("torch")
local Module = require("torch.nn.module")

local function class(name)
  return Module.class(name)
end

-- Linear ---------------------------------------------------------------------

local Linear = class("Linear")

function Linear.new(in_features, out_features, bias)
  if type(in_features) ~= "number" or type(out_features) ~= "number" then
    error("Linear(in_features, out_features, bias?)", 3)
  end
  if bias == nil then
    bias = true
  end
  local self = setmetatable(Module.new(), Linear)
  self.in_features = in_features
  self.out_features = out_features

  local bound = 1 / math.sqrt(in_features)
  local weight = torch.rand(out_features, in_features):mul(2 * bound):sub(bound)
  self:register_parameter("weight", weight)
  self.weight = self._parameters.weight

  if bias then
    local bias_tensor = torch.rand(out_features):mul(2 * bound):sub(bound)
    self:register_parameter("bias", bias_tensor)
    self.bias = self._parameters.bias
  else
    self.bias = nil
  end
  return self
end

function Linear:forward(input)
  if self.bias then
    if input:dim() == 2 then
      -- One addmm kernel instead of matmul + separate bias add.
      return self.bias:addmm(input, self.weight:t())
    elseif input:dim() > 2 then
      local sizes = input:sizes()
      local flattened = input:reshape(-1, self.in_features)
      local output = self.bias:addmm(flattened, self.weight:t())
      local output_sizes = {}
      for index = 1, #sizes - 1 do
        output_sizes[index] = sizes[index]
      end
      output_sizes[#output_sizes + 1] = self.out_features
      return output:reshape(output_sizes)
    end
    return input:matmul(self.weight:t()):add(self.bias)
  end
  return input:matmul(self.weight:t())
end

function Linear:_refresh_aliases()
  self.weight = self._parameters.weight
  self.bias = self._parameters.bias
end

function Linear:format()
  return string.format("Linear(%d -> %d, bias=%s)",
    self.in_features, self.out_features, tostring(self.bias ~= nil))
end

-- Activations ----------------------------------------------------------------

local ReLU = class("ReLU")

function ReLU.new(inplace)
  local self = setmetatable(Module.new(), ReLU)
  self.inplace = inplace and true or false
  return self
end

function ReLU:forward(input)
  if self.inplace then
    return input:relu_()
  end
  return input:relu()
end

local LeakyReLU = class("LeakyReLU")

function LeakyReLU.new(negative_slope, inplace)
  local self = setmetatable(Module.new(), LeakyReLU)
  self.negative_slope = negative_slope or 0.01
  self.inplace = inplace and true or false
  return self
end

function LeakyReLU:forward(input)
  local negative = input:mul(self.negative_slope)
  local output = input:maximum(negative)
  if self.inplace then
    input:copy_(output)
    return input
  end
  return output
end

local Sigmoid = class("Sigmoid")

function Sigmoid.new()
  return setmetatable(Module.new(), Sigmoid)
end

function Sigmoid:forward(input)
  return input:sigmoid()
end

local Tanh = class("Tanh")

function Tanh.new()
  return setmetatable(Module.new(), Tanh)
end

function Tanh:forward(input)
  return input:tanh()
end

local Softplus = class("Softplus")

function Softplus.new(beta, threshold)
  local self = setmetatable(Module.new(), Softplus)
  self.beta = beta or 1
  self.threshold = threshold or 20
  return self
end

function Softplus:forward(input)
  return input:softplus(self.beta, self.threshold)
end

function Softplus:format()
  return string.format("Softplus(beta=%g, threshold=%g)",
    self.beta, self.threshold)
end

local ELU = class("ELU")

function ELU.new(alpha, inplace)
  local self = setmetatable(Module.new(), ELU)
  self.alpha = alpha or 1
  self.inplace = inplace and true or false
  return self
end

function ELU:forward(input)
  local output = input:elu(self.alpha)
  if self.inplace then
    input:copy_(output)
    return input
  end
  return output
end

function ELU:format()
  return string.format("ELU(alpha=%g, inplace=%s)", self.alpha,
    tostring(self.inplace))
end

local SELU = class("SELU")

function SELU.new(inplace)
  local self = setmetatable(Module.new(), SELU)
  self.inplace = inplace and true or false
  return self
end

function SELU:forward(input)
  local output = input:selu()
  if self.inplace then
    input:copy_(output)
    return input
  end
  return output
end

local SiLU = class("SiLU")

function SiLU.new(inplace)
  local self = setmetatable(Module.new(), SiLU)
  self.inplace = inplace and true or false
  return self
end

function SiLU:forward(input)
  local output = input:silu()
  if self.inplace then
    input:copy_(output)
    return input
  end
  return output
end

local GELU = class("GELU")

function GELU.new(approximate)
  local self = setmetatable(Module.new(), GELU)
  self.approximate = approximate or "none"
  return self
end

function GELU:forward(input)
  return input:gelu(self.approximate)
end

function GELU:format()
  return string.format("GELU(approximate=%q)", self.approximate)
end

local Softmax = class("Softmax")

function Softmax.new(dim)
  local self = setmetatable(Module.new(), Softmax)
  self.dim = dim
  return self
end

function Softmax:forward(input)
  return input:softmax(self.dim)
end

local LogSoftmax = class("LogSoftmax")

function LogSoftmax.new(dim)
  local self = setmetatable(Module.new(), LogSoftmax)
  self.dim = dim
  return self
end

function LogSoftmax:forward(input)
  return input:log_softmax(self.dim)
end

local Identity = class("Identity")

function Identity.new()
  return setmetatable(Module.new(), Identity)
end

function Identity:forward(input)
  return input
end

-- Flatten --------------------------------------------------------------------

local Flatten = class("Flatten")

function Flatten.new(start_dim, end_dim)
  local self = setmetatable(Module.new(), Flatten)
  self.start_dim = start_dim or 2
  self.end_dim = end_dim or -1
  return self
end

function Flatten:forward(input)
  return input:flatten(self.start_dim, self.end_dim)
end

-- Sequential -----------------------------------------------------------------

local Sequential = class("Sequential")

function Sequential.new(...)
  local self = setmetatable(Module.new(), Sequential)
  local modules = { ... }
  if #modules == 1 and type(modules[1]) == "table" and modules[1].forward == nil then
    modules = modules[1]
  end
  for _, module in ipairs(modules) do
    self:add(module)
  end
  return self
end

function Sequential:add(module)
  if not (type(module) == "table" and module.forward) then
    error("Sequential:add expects a Module", 3)
  end
  local name = tostring(#self._module_order + 1)
  self._modules[name] = module
  self._module_order[#self._module_order + 1] = name
  return self
end

function Sequential:forward(input)
  local output = input
  for _, name in ipairs(self._module_order) do
    output = self._modules[name](output)
  end
  return output
end

function Sequential:format()
  local parts = {}
  for _, name in ipairs(self._module_order) do
    parts[#parts + 1] = tostring(self._modules[name])
  end
  return "Sequential(" .. table.concat(parts, ", ") .. ")"
end

-- ModuleList -----------------------------------------------------------------

local ModuleList = class("ModuleList")

function ModuleList.new(modules)
  local self = setmetatable(Module.new(), ModuleList)
  self:add(modules)
  return self
end

function ModuleList:add(module)
  if type(module) == "table" and module.forward == nil then
    for _, item in ipairs(module) do
      self:add(item)
    end
    return self
  end
  local name = tostring(#self._module_order + 1)
  self._modules[name] = module
  self._module_order[#self._module_order + 1] = name
  return self
end

function ModuleList:get(index)
  return self._modules[tostring(index)]
end

function ModuleList:forward(input)
  error("ModuleList is a container, not a network; iterate over it", 2)
end

return {
  Linear = Linear,
  ReLU = ReLU,
  LeakyReLU = LeakyReLU,
  Sigmoid = Sigmoid,
  Tanh = Tanh,
  Softplus = Softplus,
  ELU = ELU,
  SELU = SELU,
  SiLU = SiLU,
  GELU = GELU,
  Softmax = Softmax,
  LogSoftmax = LogSoftmax,
  Identity = Identity,
  Flatten = Flatten,
  Sequential = Sequential,
  ModuleList = ModuleList,
}
