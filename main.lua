--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'

local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')

local opt = opts.parse(arg)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
  local nTests = 100
  local accuracyForward, accuracyBackward = 0.0, 0.0

    -- add softmax layer to output probabilities and set the model in test mode
    model:add(nn.SoftMax())
    model:cuda()
    model:evaluate()

  for i = 1, nTests do
    local predPerVideo, correctForward, correctBackward, countForward, countBackward, countRest = valLoader:accuracyPerVideo(model)
    accuracyForward = accuracyForward + correctForward / countForward
    accuracyBackward = accuracyBackward + correctBackward / countBackward
    print(i, accuracyForward / i, accuracyBackward / i, correctForward / countForward, correctBackward / countBackward)
  end
  print(accuracyForward / nTests, accuracyBackward / nTests)
  return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestError = math.huge


local logger = optim.Logger(opt.save .. '/' .. os.date('%d-%m-%y:%H:%M') .. '.log')
logger:setNames{'epoch', 'training error', 'test error', 'training time', 'testing time', '11tr', '12tr', '13tr', '21tr', '22tr', '23tr', '31tr', '32tr', '33tr', '11va', '12va', '13va', '21va', '22va', '23va', '31va', '32va', '33va'}

for epoch = startEpoch, opt.nEpochs do
   -- train for a single epoch
   local timer = torch.Timer()
   local trainError, trainConfMat = trainer:train(epoch, trainLoader)
   local trainTime = timer:time().real

   -- run model on validation set
   timer:reset()
   local testError, testConfMat = trainer:test(epoch, valLoader)
   local testTime = timer:time().real
   -- print and save to log
   print('Train error, test error = ', trainError, testError)
   print(testConfMat)
   logger:add{epoch, trainError, testError, trainTime, testTime, trainConfMat[1][1], trainConfMat[1][2], trainConfMat[1][3], trainConfMat[2][1], trainConfMat[2][2], trainConfMat[2][3], trainConfMat[3][1], trainConfMat[3][2], trainConfMat[3][3], testConfMat[1][1], testConfMat[1][2], testConfMat[1][3], testConfMat[2][1], testConfMat[2][2], testConfMat[2][3], testConfMat[3][1], testConfMat[3][2], testConfMat[3][3]}

   local bestModel = false
   if testError < bestError then
      bestModel = true
      bestError = testError
      print(' * Best model, test error: ', bestError)
   end
   checkpoints.save(opt.nEpochs, model, trainer.optimState, bestModel, opt)
end

-- display plot
logger:style{'-', '-'}
logger:plot()

-- recover first layer filters
require 'image'

tripleNet = model.modules[1]
simpleNet = tripleNet.modules[1]
simpleFeatures = simpleNet.modules[1]
filters = simpleFeatures.modules[1].weight
nbFilters = filters:size(1)

for i = 1, nbFilters do
  image.save(opt.filters .. '/filter_' .. i .. '.jpg', image.toDisplayTensor(filters[i]))
end

-- save model
checkpoints.save(opt.nEpochs, model, trainer.optimState, bestModel, opt)
print(string.format(' * Finished, test error: ', bestError))
