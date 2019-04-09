import torch
import torch.nn as nn



class ComplicatedConvolutionalLayer(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, pad, stride = 1):
        super(ComplicatedConvolutionalLayer, self).__init__()
        
        self.conv = nn.Conv2d(inChannels, outChannels,  stride=1,
                              kernel_size=kernelSize, padding=pad, bias=False)
        self.batchNorm = nn.BatchNorm2d(outChannels)
        self.relu = nn.LeakyReLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x


class ConvolutionalLayer(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(ConvolutionalLayer, self).__init__()
        
        self.conv = nn.Conv2d(inChannels, outChannels,  stride=1,
                              kernel_size=3, padding=1, bias=False)
        self.batchNorm = nn.BatchNorm2d(outChannels)
        self.relu = nn.LeakyReLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x




class TroubleShooter(nn.Module):
    def __init__(self):
        super(TroubleShooter, self).__init__()
    
        self.features = nn.Sequential()
        
        self.features.add_module('conv1', ConvolutionalLayer(1, 32))
        self.features.add_module('conv2', ConvolutionalLayer(32, 32))
        self.features.add_module('maxPool1', nn.MaxPool2d(2,2,(1,0)))
        self.features.add_module('conv3', ConvolutionalLayer(32, 64))
        self.features.add_module('conv4', ConvolutionalLayer(64, 64))
        self.features.add_module('helper', torch.nn.ZeroPad2d((0, 1, 0, 0)))
        self.features.add_module('maxPool2', nn.MaxPool2d(2,2,0))
        self.features.add_module('conv5', ConvolutionalLayer(64, 128))
        self.features.add_module('conv6', ConvolutionalLayer(128, 128))
        self.features.add_module('conv7', ComplicatedConvolutionalLayer(128, 1, 1, 0))

    def forward(self, x):
        x = self.features(x)
        return x



