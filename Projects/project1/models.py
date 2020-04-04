## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        #CNN  Layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.conv3 = nn.Conv2d(64,128,3)
        self.conv4 = nn.Conv2d(128,256,2)
        
        #Maxpooling
        self.pool = nn.MaxPool2d(2,2)
        
        #Dropout
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.3)
        
        
        #Fully connected Linear layers
        self.fc1 = nn.Linear(36864,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500,136)
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to  
        # avoid overfitting
        ##=================================================##
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # a modified x, having gone through all the layers of your model, should be returned

        
    def forward(self, x):
        
#         print("Before C1:",x.shape)
        x = self.pool(F.relu(self.conv1(x))) #CNN -> RELU -> MAXPOOL 
#         print("C1:",x.shape)
        x = self.pool(F.relu(self.conv2(x))) #CNN -> RELU -> MAXPOOL 
#         print("C2:",x.shape)
        x = self.pool(F.relu(self.conv3(x))) #CNN -> RELU -> MAXPOOL 
#         print("C3:",x.shape)
        x = self.pool(F.relu(self.conv4(x))) #CNN -> RELU -> MAXPOOL
#         print("C4:",x.shape)
        
        
        
        x = x.view(x.size(0),-1)             #RESIZING to VECTOR / FLATTENING
#         print("After flattening:",x.shape)
            
        x = self.dropout1(F.relu(self.fc1(x)))  # Linear Operation -> RELU -> DROPOUT
#         print("FC1:",x.shape)
        x = self.dropout2(F.relu(self.fc2(x)))  # Linear Operation -> RELU -> DROPOUT
#         print("FC2:",x.shape)
        x = self.fc3(x)                         # Linear Operation 
#         print("FC3:",x.shape)
        
        return x
# model = Net()
# model