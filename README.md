# How to see time's arrow in video sequences ? 

This repo was adapted from fb.resnet.torch to the particular problem of this project. The aim is to predict whether a video is running forwards or backwards using a triplet siamese CNN. 

1. [Setup](#setup)

2. [Playing around](#playing-around)

3. [Going further](#going-further)

 
## Setup 

The dataset is available [here](http://www.robots.ox.ac.uk/~vgg/data/arrow/) and contains 

To start a training, simply run the following command in your favourite shell
```
th main.lua -nGPU 1 -nThreads 8 -nEpochs 1 -batchSize 3 -LR 0.00001 -garbageClass true
```
To evaluate the latest model you have trained, simply run
```
th main.lua -testOnly true -resume <pathToCheckpointFolder>
```

## Playing around

Change the dataset 

## Going further

You 
