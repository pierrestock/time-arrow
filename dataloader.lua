--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local videoSplits = require 'sampling/video-splits'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val'} do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
   end

   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         --torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
   self.threads = threads
   self.batchSize = math.floor(opt.batchSize / self.nCrops)
   self.dataset = dataset
   self.preprocess = dataset:preprocess()
   self:shuffle(dataset, opt, split)
end

function DataLoader:size()
   return math.ceil(self.tripletCount / self.batchSize)
end

function DataLoader:accuracyPerVideo(model)
    local videoNames = self.dataset.imageInfo.videoNames
    local imageSize = {3, 224, 224}
    local predPerVideo = {}
    local correctForward = 0
    local correctBackward = 0
    local countRest = 0
    local countForward = 0
    local countBackward = 0
    local currentVideo = ''
    local i = 0
    local counter = 0

    -- for every video, compute predPerVideo
    while counter < 36 do
        while videoNames[i] == currentVideo or videoNames[i] ~= videoNames[i + 50] do
            i = i + 1
        end
        currentVideo = videoNames[i]
        counter = counter + 1
        local probsSumForward = torch.zeros(3):float()
        local probsSumBackward = torch.zeros(3):float()
        -- sample 10 forward triplets
        local isBackward = torch.random(0, 1)
        for j = 1, 10 do
            local seed = i + torch.random(1, 50)
            local idx = {seed, seed + 25, seed + 50}
            local tripletForward = torch.FloatTensor(3, table.unpack(imageSize))
            local tripletBackward = torch.FloatTensor(3, table.unpack(imageSize))
            for k = 1, 3 do
              local sample = self.dataset:get(idx[k])
              local input = self.preprocess(sample.input)
              -- save images
              --require 'image'
              --local path = '/home/ubuntu/object/data/samples/sample_'
              --image.save(path .. videoNames[i] .. k ..'.jpg', sample.input)
              -- end save images
              if isBackward == 0 then
                tripletForward[k]:copy(input)
                tripletBackward[3 - k + 1]:copy(input)
              else
                tripletForward[3 - k + 1]:copy(input)
                tripletBackward[k]:copy(input)
              end
            end
            outputForward = model:forward(tripletForward:cuda()):float()
            outputBackward = model:forward(tripletBackward:cuda()):float()
            probsSumForward = probsSumForward + outputForward
            probsSumBackward= probsSumBackward + outputBackward
        end
        stats = {}
        stats.F = probsSumForward / 10
        stats.B = probsSumBackward / 10
        stats.T = videoSplits[videoNames[i]]
        stats.C = isBackward
        predPerVideo[i] = stats
        if videoSplits[videoNames[i]] == 'action' then
          if (probsSumForward[1] + probsSumBackward[2] > probsSumForward[2] + probsSumBackward[1]) and  isBackward == 0 then
            correctForward = correctForward + 1
          elseif (probsSumForward[1] + probsSumBackward[2] > probsSumForward[2] + probsSumBackward[1]) and isBackward == 1 then
            correctBackward = correctBackward + 1
          end
          countForward = countForward + (1 - isBackward)
          countBackward = countBackward + isBackward
        else
          countRest = countRest + 1
        end
    end
    return predPerVideo, correctForward, correctBackward, countForward, countBackward, countRest
end

function DataLoader:shuffle(dataset, opt, split)
    -- variables
    local imageNames = dataset.imageInfo.imageNames
    local videoNames = dataset.imageInfo.videoNames
    local size = #imageNames
    local triplets = {}
    local targets = {}
    local counter = 0
    local counter1 = 0
    local counter2 = 0
    local counter3 = 0
    -- loop
    while counter < size do
        local i = torch.random(1, size - 49)
        if videoNames[i] == videoNames[i  + 49] then
            -- not much happens in video, class 3
            if videoSplits[videoNames[i]] == opt.garbageClass then
                local shift = torch.random(1,5)
                for k = 20, 29 do
                    table.insert(triplets, {i, i + k + shift, i + 49})
                    table.insert(targets, 3)
                end
                counter3 = counter3 + 10
            else
                -- forward or backward samples
                local seed = torch.random(1,2)
                local shift = torch.random(1,5)
                if seed % 2 == 0 then
                    -- forward sample, class 1
                    for k = 20, 29 do
                        table.insert(triplets, {i, i + k + shift, i + 49})
                        table.insert(targets, 1)
                    end
                    counter1 = counter1 + 10
                else
                    -- backward sample, class 2
                    for k = 20, 29 do
                        table.insert(triplets, {i + 49, i + k + shift, i})
                        table.insert(targets, 2)
                    end
                    counter2 = counter2 + 10
                end
            end
            counter = counter + 10
            -- save some training/test examples
            require 'image'
            if opt.displaySamples and i % 100 == 0 then
                local path = opt.samples .. '/sample_'
                image.save(path .. i .. '1.jpg', dataset:get(i + 0).input)
                image.save(path .. i .. '2.jpg', dataset:get(i + 2).input)
                image.save(path .. i .. '3.jpg', dataset:get(i + 5).input)
            end
        end
    end
    self.tripletCount = counter
    print(split, counter, counter1, counter2, counter3)
    -- shuffle
    shuffle = torch.randperm(counter):long()
    self.triplets = torch.Tensor(triplets):index(1, shuffle)
    self.targets = torch.Tensor(targets):index(1, shuffle)
end

function DataLoader:run()
   -- shuffle
   shuffle = torch.randperm(self.triplets:size(1)):long()
   self.triplets = self.triplets:index(1, shuffle)
   self.targets = self.targets:index(1, shuffle)
   local threads = self.threads
   local batchSize = self.batchSize
   local perm = self.triplets
   local perm_targets = self.targets
   local size = perm:size(1)
   local idx, sample = 1, nil
   local function enqueue()
      while idx <= (size - batchSize) and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         local targets = perm_targets:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices, targets, nCrops)
               local sz = indices:size(1)
               local imageSize = {3, 224, 224}
               local batch = torch.FloatTensor(sz, 3, table.unpack(imageSize))
               local target = torch.IntTensor(sz)
               for i, idx in ipairs(indices:totable()) do
                  hsplit = torch.random(0, 1)
                  for j = 1, 3 do
                    local sample = _G.dataset:get(idx[j])
                    local input = _G.preprocess(sample.input, hsplit)
                    batch[i][j]:copy(input)
                  end
                  target[i] = targets[i]
               end
               collectgarbage()
               return {
                  input = batch,
                  target = target,
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            targets,
            self.nCrops
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
