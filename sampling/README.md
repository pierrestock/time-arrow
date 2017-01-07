# Sampling

Here is a brief description of the files:

* __get-optical-flow.m__ Calculates the average magnitude of the optical flow per video in order to separate the training and test set into 2 categories: *rest* and *action* (50-50 split)
* __video-splits.lua__ The video splits into 'rest' or 'action' as calculated in the previous file. 
