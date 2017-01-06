require 'nn'
require 'cunn'
require 'cudnn'

local function createModel(opt)
    -- load model to replicate
    if opt.pretrained then
        print('Using pretrained model')
        -- loading pretrained model features
        features = torch.load(opt.pretrainedPath .. '/'  .. opt.netType .. '/' .. 'features.t7'):unpack()
        features:add(cudnn.SpatialMaxPooling(4,4,2,2))

        -- adding top raw linear layers
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

    else
        print('Using raw model')
        model = require('models/' .. opt.netType)
    end

    -- the entire model will be made of 3 parts {shared, concat, classifier}
    net = nn.Sequential()

    -- shared part + concatenate part
    shared = nn.Parallel(1, 1)

    model:add(nn.View(4096 ,1)) --4096

    fb1 = model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
    fb2 = model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
    fb3 = model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')

    shared:add(fb1)
    shared:add(fb2)
    shared:add(fb3)

    -- classifier part
    classifier = nn.Sequential()
    classifier:add(nn.View(3 * 4096))
    classifier:add(nn.Linear(3 * 4096, opt.nClasses))

    -- build entire model
    net:add(shared)
    net:add(classifier)

    -- transfer it to GPU
    net:cuda()

    return net
end

  return createModel
