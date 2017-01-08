--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 Triplet CNN Training script')
   cmd:text('Adapted from https://github.com/facebook/fb.resnet.torch/')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',            '/home/ubuntu/object/data/youtube-reduce',    'Path to dataset')
   cmd:option('-dataset',         'youtube',                                    'Name of dataset ')
   cmd:option('-manualSeed',      0,                                            'Manually set RNG seed')
   cmd:option('-nGPU',            1,                                            'Number of GPUs to use by default')
   cmd:option('-backend',         'cudnn',                                      'Options: cudnn | cunn')
   cmd:option('-cudnn',           'fastest',                                    'Options: fastest | default | deterministic')
   cmd:option('-gen',             '/home/ubuntu/object/data/gen',               'Path to save generated files')
   cmd:option('- filters',        '/home/ubuntu/object/data/filters',           'Path to save 1st layer filters')
   cmd:option('- samples',        '/home/ubuntu/object/data/samples',           'Path to save some random training samples')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        8,                                            'Number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         1,                                            'Number of total epochs to run')
   cmd:option('-epochNumber',     1,                                            'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       3,                                            'Mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',        'false',                                      'Calculate per video accuracy')
   cmd:option('-displaySamples',  'false',                                      'Display some training/testing samples')
   cmd:option('-garbageClass',    'false',                                      'Use of a third class (see doc)')
   cmd:option('-retrain',         'none',                                       'Retrain model')
   cmd:option('-tenCrop',         'false',                                      '10-crop testing (do not use here)')
   ------------- Checkpointing options ---------------
   cmd:option('-save',            '/home/ubuntu/object/data/checkpoints',       'Directory in which to save checkpoints')
   cmd:option('-resume',          'false',                                      'Resume from the latest checkpoint in this directory')
   ---------- Optimization options ----------------------
   cmd:option('-LR',              0.0001,                                       'Initial learning rate')
   cmd:option('-momentum',        0.9,                                          'Momentum')
   cmd:option('-weightDecay',     1e-4,                                         'Weight decay')
   ---------- Model options ----------------------------------
   cmd:option('-netType',         'alexnet',                                    'Options: alexnet | other')
   cmd:option('-pretrained',      'false',                                      'Options: alexnet | other')
   cmd:option('-pretrainedPath',  '/home/ubuntu/object/data/pretrained',        'Path to pretrained models')
   cmd:option('-optimState',      'none',                                       'Path to an optimState to reload from')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',  'false',                                      'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',          'false',                                      'Use optnet to reduce memory usage')
   cmd:option('-resetClassifier', 'false',                                      'Reset the fully connected layer for fine-tuning')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'
   opt.pretrained = opt.pretrained ~= 'false'
   opt.displaySamples = opt.displaySamples ~= 'false'

   if opt.resume ~= 'false' then
        opt.resume = opt.save
   end

   if opt.garbageClass == 'false' then
        opt.garbageClass = 'no'
        opt.nClasses = 2
   else
        opt.garbageClass = 'rest'
        opt.nClasses = 3
   end

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
