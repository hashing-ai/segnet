# segnet : An Encoder-Decoder Architecture for Segmentation

- Based on the [paper](https://arxiv.org/pdf/1511.00561.pdf)

### To-Do 

- import data
- transform data
- create batches
- create model
- run model
- evaluate


## some things about the model

- Major Layers/Block in the architecture:
    1. An **ENCODER** Network 
    2. A **DECODER** Network
    3. A pixelwise classification layer
- The encoder network consists of 13 convolutional layers which correspond to the first 13 layers from the VGG network.
    - But, wait VGG is for object classification and not for segmentation.
    - It still works, as the weights trained on a large image classification dataset are a great start to initialize the training process on. 
    - they also discarded the fully connected layers in favour of retaining higher resolution feature maps at the deepest encoder output.
    - Also reducing the number of paramaters in the SegNet encoder netwrok significantly from 134M to 14.7M.
- Each of the encoder layer has a corresponding decoder layer and hence the decoder network has 13 layers. 
- The final decoder output is fed to a multi-class soft-max calssifier to produce class probabilities for each pixel independently.


