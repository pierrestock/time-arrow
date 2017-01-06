require 'nn'
require 'cunn'
require 'cudnn'

SpatialConvolution = cudnn.SpatialConvolution
SpatialMaxPooling = cudnn.SpatialMaxPooling
ReLU = cudnn.ReLU

features = nn.Sequential()
features:add(SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
features:add(ReLU(true))
features:add(SpatialMaxPooling(3,3,2,2))                   -- 55 -> 27
features:add(SpatialConvolution(64,192,5,5,1,1,2,2))       -- 27 -> 27
features:add(ReLU(true))
features:add(SpatialMaxPooling(3,3,2,2))                   -- 27 -> 13
features:add(SpatialConvolution(192,384,3,3,1,1,1,1))      -- 13 -> 13
features:add(ReLU(true))
features:add(SpatialConvolution(384,256,3,3,1,1,1,1))      -- 13 -> 13
features:add(ReLU(true))
features:add(SpatialConvolution(256,256,3,3,1,1,1,1))      -- 13 -> 13
features:add(ReLU(true))
features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

linear = nn.Sequential()
linear:add(nn.View(256*6*6))
--linear:add(nn.Dropout(0.5))
linear:add(nn.Linear(256*6*6, 4096))
linear:add(nn.Threshold(0, 1e-6))
--linear:add(nn.Dropout(0.5))
linear:add(nn.Linear(4096, 4096))
linear:add(nn.Threshold(0, 1e-6))

model = nn.Sequential()

model:add(features):add(linear)

return model
