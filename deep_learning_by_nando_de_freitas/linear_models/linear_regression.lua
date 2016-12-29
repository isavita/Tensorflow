require 'torch'
require 'optim'
require 'nn'

-- Set the to a text file
logger = optim.Logger('cost_log.txt')

-- Data {corn produced, fertilizer, insecticide}
data = torch.Tensor{
  {40,  6,  4},
  {44, 10,  4},
  {46, 12,  5},
  {48, 14,  7},
  {52, 16,  9},
  {58, 18, 12},
  {60, 22, 14},
  {68, 24, 20},
  {74, 26, 21},
  {80, 32, 24}
}

-- Define the model
model = nn.Sequential()
ninputs = 2
noutputs = 1
model:add(nn.Linear(ninputs, noutputs))

-- Define the cost function (mean square error)
criterion = nn.MSECriterion()

-- Train the model with stochastic gradient descent
w, dc_dw = model:getParameters()

feval = function(w_new)
  if w ~= w_new then
    w:copy(w_new)
  end

  -- Select a new training sample

  _nidx_ = (_nidx_ or 0) + 1
  if _nidx_ > (#data)[1] then _nidx_ = 1 end

  local sample = data[_nidx_]
  local target = sample[{ {1} }]
  local inputs = sample[{ {2, 3} }]

  -- Reset gradient
  dc_dw:zero()

  -- Evaluate the cost function with respect to w, for that sample
  local cost_w =  criterion:forward(model:forward(inputs), target)
  model:backward(inputs, criterion:backward(model.output, target))
  
  -- Return cost(w) and dcost/dw
  return cost_w, dc_dw 
end

-- Training parameters:
-- 1) learning reate - the size of the step taken at each stochastic estimateof the gradient
-- 2) weight decay - regularize the solution (L2 regularization)
-- 3) momentum term - average step over time
-- 4) learning rate decay - let the algorithm converge more precisely

sgd_params = {
  learningRate = 1e-3,
  learningRateDecay = 1e-4,
  weightDecay = 0,
  momentum = 0
}

for i = 1, 1e4 do
  current_cost = 0
  for i = 1, (#data)[1] do
    _, fs = optim.sgd(feval, w, sgd_params)
    current_cost = current_cost + fs[1]
  end
  
  -- Report average error on epoch
  current_cost = current_cost / (#data)[1]
  print('current cost = ' .. current_cost)

  logger:add{['training error'] = current_cost}
  logger:style{['training error'] = '-'}
  logger:plot()
end

-- Test the trained model
-- The text solves the model exactly using matrix techniques and determines
-- corn = 31.98 + 0.65 * fertilizer + 1.11 * insecticides
-- We compare our approximate results with the text's results.
text = {40.32, 42.92, 45.33, 48.85, 52.37, 57, 61.82, 69.78, 72.19, 79.42}
print('id  approx   text')
for i = 1,(#data)[1] do
   local myPrediction = model:forward(data[i][{{2,3}}])
   print(string.format("%2d  %6.2f %6.2f", i, myPrediction[1], text[i]))
end