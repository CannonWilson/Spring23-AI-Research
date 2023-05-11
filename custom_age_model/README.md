This folder contains the code to train a young/old
classifier that uses the following architecture
I came up with:

### Blocks 1-4
2D Convolution -> Relu -> BatchNorm -> 
2D Conv -> Relu -> Max Pool -> BatchNorm

### Fully Connected (FC) Layers
FC1 -> Relu -> FC2 -> Softmax

This architecture provided decent enough 
accuracy (~70% on validation set) while training
and predicting *significantly* faster than
ResNet, which is obviously much larger than 
this CNN. This speed made it ideal for 
the early experiments I was running. 

This model is no longer being used and may contain
bugs that prevent it from working as the model
in the other files. The current models being used
in all experiments can be found in the 
'./resnet_models' folder. I may revive this model
or a similar one down the road in order to test 
our method using a simpler classifier.