--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local errSum, lossSum = 0.0, 0.0
   local confMatSum = torch.zeros(3,3)
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      local err, confMat = self:computeScore(output, sample.target, 1)
      errSum = errSum + err * batchSize
      confMatSum = confMatSum + confMat
      lossSum = lossSum + loss * batchSize

      N = N + batchSize

      print((' | Epoch: [%d][%d/%d]  Time %.3f  Loss %1.4f  Err %7.3f (%7.3f)'):format(
         epoch, n, trainSize, timer:time().real, loss, err, errSum / N))
      print(confMatSum / N)

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
   end

   return errSum / N, confMatSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local errSum = 0.0
   local confMatSum = torch.zeros(3,3)
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run() do

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1) / nCrops
      local loss = self.criterion:forward(self.model.output, self.target)

      local err, confMat = self:computeScore(output, sample.target, nCrops)
      errSum = errSum + err * batchSize
      confMatSum = confMatSum + confMat

      N = N + batchSize
      print((' | Test: [%d][%d/%d]  Time %.3f  Err %7.3f (%7.3f)'):format(
         epoch, n, size, timer:time().real, err, errSum / N))
      print(confMatSum / N)

      timer:reset()
   end
   self.model:training()

   print((' * Finished epoch # %d  Err %7.3f \n'):format(
      epoch, errSum / N))

   return errSum / N, confMatSum / N
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- confusion matrix
   local confMat = torch.zeros(3, 3)

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Find predictions of class 1
   local class1 = predictions:eq(1 * torch.ones(batchSize, 1):long():expandAs(output)):narrow(2, 1, 1)

   local pred1 = target:long():eq(1 * torch.ones(batchSize, 1):long())
   local pred2 = target:long():eq(2 * torch.ones(batchSize, 1):long())
   local pred3 = target:long():eq(3 * torch.ones(batchSize, 1):long())

   confMat[1][1] = torch.cmul(class1, pred1):sum()
   confMat[1][2] = torch.cmul(class1, pred2):sum()
   confMat[1][3] = torch.cmul(class1, pred3):sum()

   -- Find predictions of class 2
   local class2 = predictions:eq(2 * torch.ones(batchSize, 1):long():expandAs(output)):narrow(2, 1, 1)

   local pred1 = target:long():eq(1 * torch.ones(batchSize, 1):long())
   local pred2 = target:long():eq(2 * torch.ones(batchSize, 1):long())
   local pred3 = target:long():eq(3 * torch.ones(batchSize, 1):long())

   confMat[2][1] = torch.cmul(class2, pred1):sum()
   confMat[2][2] = torch.cmul(class2, pred2):sum()
   confMat[2][3] = torch.cmul(class2, pred3):sum()

   -- Find predictions of class 3
   local class3 = predictions:eq(3 * torch.ones(batchSize, 1):long():expandAs(output)):narrow(2, 1, 1)

   local pred1 = target:long():eq(1 * torch.ones(batchSize, 1):long())
   local pred2 = target:long():eq(2 * torch.ones(batchSize, 1):long())
   local pred3 = target:long():eq(3 * torch.ones(batchSize, 1):long())

   confMat[3][1] = torch.cmul(class3, pred1):sum()
   confMat[3][2] = torch.cmul(class3, pred2):sum()
   confMat[3][3] = torch.cmul(class3, pred3):sum()
   -- prediction error
   local err = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   return err, confMat
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor()or torch.CudaTensor())
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   return self.opt.LR / epoch
end

return M.Trainer
