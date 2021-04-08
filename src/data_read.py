# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torchvision
import torch
import torch.utils.data as data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import csv

class MyDataset(data.Dataset):
    def __init__(self, root, usage='train', resize=227, test_radio=0.3):
        super(MyDataset,self).__init__()
        self.root=root
        self.usage=usage
        self.image_name=[]
        self.labels=[]
        self.resize= resize

        with open(self.root+'data/train_labeled_Elbow.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for line in reader:
                for pic in os.listdir(root+line[0]):
                    self.image_name.append(root+line[0]+pic)
                    self.labels.append(line[1])
        self.num_classes=2
        temp= np.array([self.image_name, self.labels])
        temp= temp.transpose()
        np.random.shuffle(temp)
        self.image_name_train=temp[:,0]
        self.labels_train=temp[:,1]
        
        test_image=[]
        test_labels=[]
        with open(self.root+'data/valid_labeled_Elbow.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for line in reader:
                for pic in os.listdir(root+line[0]):
                    test_image.append(root+line[0]+pic)
                    test_labels.append(line[1])
        temp= np.array([test_image, test_labels])
        temp= temp.transpose()
        np.random.shuffle(temp)
        self.image_name_test=temp[:,0]
        self.labels_test=temp[:,1]
        onehot= OneHotEncoder(sparse=False, categories='auto')
        integer= LabelEncoder()
        self.labels_train=np.array(self.labels_train)
        self.labels_test= np.array(self.labels_test)
        self.labels_train= self.labels_train.reshape(len(self.labels_train),1)
        self.labels_test= self.labels_test.reshape(len(self.labels_test),1)
        self.labels_train= integer.fit_transform(self.labels_train)
        self.labels_test= integer.fit_transform(self.labels_test)

    def __getitem__(self, index):
        """
        :param index:
            index(int): Index
        :return:
            tuple: (image, target) where target is index of the target class.
        """
        if self.usage=='train':
            img_name= self.image_name_train[index]
            img= Image.open(img_name).convert('RGB')
            target= self.labels_train[index]
            transforms1 = transforms.Compose([
                          transforms.Resize(self.resize),
                          transforms.ToTensor()])
            img= transforms1(img)
            return img, target

        if self.usage=='test':
            img_name = self.image_name_test[index]
            img = Image.open(img_name).convert('RGB')
            target = self.labels_test[index]
            transforms1 = transforms.Compose([
                transforms.Resize(self.resize),
                transforms.ToTensor()])
            img = transforms1(img)
            return img, target

    def __len__(self):
        if self.usage=='train':
            return len(self.labels_train)
        if self.usage=='test':
            return len(self.labels_test)

def loadtraindata(root, batch_size, resize):
    dataset= MyDataset(root,usage='train',resize=resize)
    print("total pic number: ", len(dataset.image_name))
    print("total train number: ", len(dataset.image_name_train))
    print("total test number: ", len(dataset.image_name_test))
    trainloader= torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    return trainloader

def loadtestdata(root, batch_size, resize):
    dataset= MyDataset(root,usage='test',resize=resize)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16,
                                              pin_memory=True)
    return testloader


def loaddatafromfolder():
    path=r"/home/hzq/SkeletonDiag/"
    trainset= torchvision.datasets.ImageFolder(path, transform=transforms.Compose([transforms.Resize((227, 227)),transforms.ToTensor()]))
    trainsetforVGG16=torchvision.datasets.ImageFolder(path, transform=transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((.5,.5,.5),(.5,.5,.5))]
    ))
    trainloader= torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=16)
    trainloader_VGG = torch.utils.data.DataLoader(trainsetforVGG16, batch_size=100, shuffle=True, num_workers=16)
    return trainloader

def loadtestdatafromfolder():
    path = r"/home/hzq/SkeletonDiag/"
    trainset = torchvision.datasets.ImageFolder(path, transform=transforms.Compose(
        [transforms.Resize((227, 227)), transforms.ToTensor()]))
    trainsetforVGG16 = torchvision.datasets.ImageFolder(path, transform=transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
    ))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=16)
    trainloader_VGG = torch.utils.data.DataLoader(trainsetforVGG16, batch_size=100, shuffle=True, num_workers=16)
    return trainloader


root=r'/home/hzq/SkeletonDiag/'
dataset=MyDataset(root)

# a=[1,2,3]
# b=['a','b','c']
# temp=np.array([a,b])
# print(temp)
# print(temp[1,:])



# class Data_loader():
#     def __init__(self, file_path):
#         self.file_path= file_path
#
#     def get_files(self):
#         class_train= []
#         label_train= []
#         for first_category in os.listdir(self.file_path):
#             for second_category in os.listdir(self.file_path+'/'+first_category):
#                 for third_category in os.listdir(self.file_path+'/'+first_category+'/'+second_category):
#                     for pic in os.listdir(self.file_path+'/'+first_category+'/'+second_category+'/'+third_category+ '/Image_withrect'):
#                         class_train.append(self.file_path+'/'+first_category+'/'+second_category+'/'+third_category+ '/Image_withrect''/'+pic)
#                         label_train.append(third_category)
#         dict_label=set(label_train)
#         self.totall_num= len(class_train)
#         self.label2indice= dict((c, i) for i, c in enumerate(dict_label))
#         label_train= [self.label2indice[c] for c in label_train]
#         print("dic_label's shape:", np.shape(dict_label))
#         temp= np.array([class_train, label_train])
#         temp= temp.transpose()# 转置
#         # shuffle the samples
#         np.random.shuffle(temp)
#         image_list= list(temp[:, 0])
#         label_list= list(temp[:, 1])
#         x_train, x_test, y_train, y_test= train_test_split(image_list, label_list, test_size=0.3, random_state=0)
#         return x_train, x_test, y_train, y_test
#
#     def get_batches(self, image_path, label, resize, batch_size):
#         image_batch=[]
#         label_batch=[]
#         batch_num=1
#         if len(image_path)%batch_size==0:
#             batch_num=int(len(image_path)/batch_size)
#         else:
#             batch_num=int((len(image_path)/batch_size)+1)
#             total_num=int(batch_num*batch_size)
#             print("total_num:",total_num)
#             print("len(image_path):",len(image_path))
#             for i in range(total_num-len(image_path)):
#                 index = np.random.randint(0, len(image_path))
#                 image_path.append(image_path[index])
#                 label.append(label[index])
#         # for i in range(batch_size):
#         #     index=batch_size*batch_index+i
#         #     image=Image.open(image_path[index]).convert('RGB')
#         #     transforms1= transforms.Compose([
#         #         transforms.Scale(resize),
#         #         transforms.ToTensor()
#         #     ])
#         #     image= transforms1(image)
#         #     image_batch.append(image)
#         #     label_batch.append(label[index])
#
#         for i in range(batch_size*batch_num):
#             print("(%d,%d)"%(i, batch_num*batch_size))
#             image=Image.open(image_path[i]).convert('RGB')
#             # transforms1= transforms.Compose([
#             #     transforms.Resize(resize),
#             #     transforms.ToTensor()
#             # ])
#             transforms1=transforms.Resize(resize)
#             transforms2=transforms.ToTensor()
#             image= transforms1(image)
#             image_batch.append(np.array(image))
#             label_batch.append(label[i])
#         image_batch= np.reshape(image_batch, (batch_num, -1))
#         label_batch= np.reshape(label_batch, (batch_num, -1))
#         print("image batch's shape", np.shape(image_batch))
#         image_batch= torch.LongTensor(image_batch)
#         label_batch= torch.LongTensor(label_batch)
#         return image_batch, label_batch, batch_num
#
#     def get_num_classes(self):
#         return self.num_classes




