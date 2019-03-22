## TODO: define the convolutional neural network architecture

import torch
import torch.cuda
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def gpu(x):
    return x.cuda() if torch.cuda.is_available() else x

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv1a = nn.Conv2d(1, 32, 7,3,1) #(224-7+2)/3 +1 =74
        self.conv1b = nn.Conv2d(32, 64,7,3,1) #(74-7+2)/3 +1 = 24
        self.conv2a = nn.Conv2d(64,128,5,3,1) #(24-5+2)/3+1=8
        self.conv2b = nn.Conv2d(128,256,3) #6
        self.conv3a = nn.Conv2d(256,512,3) #4
        
        self.fc1 = nn.Linear(512*4*4,1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        act = F.relu
        
        x = act(self.conv1a(x))
        x = act(self.conv1b(x))
        
        
        x = act(self.conv2a(x))
        x = act(self.conv2b(x))
        
        x = act(self.conv3a(x))

        x = x.view(x.size(0),-1)
        x = act(self.fc1(x))
        x = self.drop1(x)
        x = act(self.fc2(x))
        x = self.drop1(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
