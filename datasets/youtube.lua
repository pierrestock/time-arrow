local image = require 'image'
local t = require 'datasets/transforms'

local M = {}
local Dataset = torch.class('Dataset', M)

function Dataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.path = opt.data .. '/' .. split
   self.hsplit = 0
   self.cropId = 1
end

function Dataset:get(i)
   local imageName = self.imageInfo.imageNames[i]
   local videoName = self.imageInfo.videoNames[i]

   local image = self:_loadImage(self.path .. '/' .. videoName .. '/' .. imageName)

   return {
      input = image,
      target = 1,
   }
end

function Dataset:_loadImage(path)
    return image.load(path, 3, 'float')
end

function Dataset:size()
   return #self.imageInfo.imageNames
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

function Dataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         --t.TenCrop(224, cropId),
         t.CenterCrop(224),
         t.ColorNormalize(meanstd),
         --t.HorizontalFlip(hsplit),
      }
   elseif self.split == 'val' then
      return t.Compose{
         t.CenterCrop(224),
         t.ColorNormalize(meanstd),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.Dataset
