# Datasets

Here is a brief description of the files:

* __resize.lua__ Resizes the dataset *in place* to 256x256 images
* __transforms.lua__ All image preprocessing transormations you may want to perform
* __youtube-gen.lua__ Generates files to describe the dataset (here *youtube*) containing folder and imagenames (the dataset is split into one __train__ and one __val__ folder, each of them contaning folders corresponding to videos). This is automatically ran only once.  
* __youtube.lua__ Dataset class, with *get* and *preprocess* methods. 
* __init.lua__ Puts everything together
