# How to see time's arrow in video sequences ? 

This repo was adapted from fb.resnet.torch to the particular problem of this project. The aim is to predict whether a video is running forwards or backwards using a triplet siamese CNN. 

1. [Setup](#setup)

2. [References](#references)

 
## Setup 

The dataset is available [here](http://www.robots.ox.ac.uk/~vgg/data/arrow/) and contains 180 videos of ~200 frames each. It was split into 144 training videos and 36 test videos. 

Here is a brief description of the files:

* __opts.lua__ All the options you can set (all have preset values and descriptions)
* __checkpoints.lua__ Handles checkpoints saving and retrieving
* __dataloader.lua__ Multi-threader dataloader
* __train.lua__ Handles training, test, score computation
* __main.lua__ Puts everything together

To start a training, simply run the following command in your favourite shell
```
th main.lua -garbageClass true
```
To evaluate the latest model you have trained, simply run
```
th main.lua -testOnly true -resume true
```

## References

* L. Pickup et al., Seeing the arrow of time, CVPR 2014 
* I. Misra, et al., Shuffle and Learn: Unsupervised Learning using Temporal Order Verification, ECCV 2016 
