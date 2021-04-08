# -*- coding: utf-8 -*-

import data_read
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Alexnet(nn.Module):
    def __init__(self, num_classes):
        self.num_classes= num_classes
        super(Alexnet, self).__init__()
        self.norm  = nn.LocalResponseNorm(size=5)
        self.drop  = nn.Dropout()
        self.pool  = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.full6 = nn.Linear(9216, 4096)
        self.full7 = nn.Linear(4096, 4096)
        self.full8 = nn.Linear(4096, self.num_classes)



    # 定义前向传播过程，输入为inputs
    def forward(self, inputs):
        batch_size= inputs.size(0) # 227*403*3
        x= self.conv1(inputs) # 55*99*96
        x= F.relu(x)
        x= self.pool(x) # 27*49*96
        x= self.norm(x)
        x= self.conv2(x) # 27*49*256
        x= F.relu(x)
        x= self.pool(x) # 13*24*256
        x= self.norm(x)
        x= self.conv3(x) # 13*24*384
        x= F.relu(x)
        x= self.conv4(x) # 13*24*384
        x= F.relu(x)
        x= self.conv5(x) # 13*24*256
        x= F.relu(x)
        x= self.pool(x) # 6*6*256
        x= self.full6(x.view(batch_size,-1)) # 4096*1
        x= F.relu(x)
        x= self.drop(x)
        x= self.full7(x) # 4096*1
        x= F.relu(x)
        x= self.drop(x)
        outputs= self.full8(x) # num_classes*1
        return outputs

    def predict(self, inputs, temperature=1.):
        inputs = torch.LongTensor(inputs)
        inputs = torch.reshape(inputs, (self.seq_size, 1))
        inputs = torch.zeros(self.seq_size, self.num_chars).scatter_(1, inputs, 1)  # one-hot encoding
        inputs = torch.reshape(inputs, (1, self.seq_size, self.num_chars))
        x = self.embedding(inputs.long())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        predicts = self.fc2(x)
        prob = F.softmax(predicts/ temperature).detach().numpy()
        return np.array([np.random.choice(self.num_chars, p=prob[0, :])])

class VGG16net(nn.Module):
    def __init__(self, num_classes):
        super(VGG16net, self).__init__()
        net= models.vgg16(pretrained=True)
        net.classifier= nn.Sequential()
        self.features= net
        self.classifier= nn.Sequential(
            nn.Linear(512*7*7,512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512,128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x= self.features(x)
        x= x.view(x.size(0), -1)
        outputs= self.classifier(x)
        return outputs

