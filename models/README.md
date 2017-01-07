# Datasets

Here is a brief description of the files:

* __alexnet.lua__ Standard AlexNet CNN with dropout
* __replicate.lua__ Stacks three networks in paralell that share all their parameters (the network is specified by its file, here AlexNet for example). Adds a classifier on the top of the 3 concatenated output features from the three siamese CNNs. Note the pretrained part which requires a bit of adaptation.  
* __init.lua__ Puts everything together

You can add as many models as you want to test them!
