# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import time
import data_read
from torch.autograd import Variable
from tqdm import tqdm
import Net_structure

device = torch.device("cpu")
path= "/home/hzq/SkeletonDiag/"
batch_size= 20
resize=(227,227)

########load test dataset
testdataloader= data_read.loadtestdata(path,batch_size, resize)


if __name__ == "__main__":
    net = Net_structure.VGG16net(2).to(device)
    net.load_state_dict(torch.load(path+'model/Wrist_VGG16_160.pt',map_location=device))
    # 训练
    start_time=time.time()
    test_acc=0
    for images, labels in tqdm(testdataloader):
        images, labels= Variable(images).to(device), Variable(labels).to(device)
        outputs = net(images)
        labels = labels.long()
        pred = torch.max(outputs, 1)[1]
        train_correct = (pred == labels).sum()
        test_acc += float(train_correct.data.item())
    end_time = time.time()
    print('Test data Acc: {:.6f}, cost time: {:.2f}'.format(test_acc / (len(testdataloader.dataset)), end_time-start_time))