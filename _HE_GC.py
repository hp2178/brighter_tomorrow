#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

import torch.nn as nn


# In[2]:


import cv2


# In[3]:


import torchvision.models as models
torch.manual_seed(3)


# In[4]:


# defining a model

model = models.resnet18(num_classes=11)


# In[5]:


device = torch.device("cuda")


# In[6]:


model.to(device)


# In[7]:


class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)

        x = self.avgpool(x)
        print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        print(x.shape)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


# In[8]:


net = ResNet50()


# In[9]:


new = torch.randn((32, 3, 96, 96))


# In[10]:


out = net(new)


# In[11]:


net


# In[12]:


from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image
from torchvision import transforms
import matplotlib.image as mpimg

import torchvision
import numpy as np
import cv2
from matplotlib import pyplot as plt


# In[13]:


import os
import zipfile
path = 'Frames_PropSplit/'
label = 0
for folder_name in os.listdir(path):
    if 'DS_Store' in folder_name:
        continue
    folder_name = folder_name + "/_test/"
    print(folder_name)
    print(label)
    cnt = 0
    for file_name in os.listdir(path + folder_name):
        if 'DS_Store' in file_name:
            continue
        true_file = folder_name + file_name
        if (cnt == 1):
            print(true_file)
        cnt = cnt + 1
        #images.append(true_file)
        #labels.append(label)
    print(cnt)
    label = label + 1


# In[14]:


# NEW CUSTOM DATA LOADER WITH PROPER SPLIT
class CustomDatasetTrain(Dataset):
    def __init__(self, path, transform=None):
        
        images = []
        labels = []
        label = 0
        
        self.root = path
        
        for folder_name in os.listdir(self.root):
            if 'DS_Store' in folder_name:
                continue
            self.folder_name = folder_name + "/_train/"
            for file_name in os.listdir(self.root + self.folder_name):
                if 'DS_Store' in file_name:
                    continue
                true_file = self.folder_name + file_name
                #print(true_file)
                images.append(true_file)
                labels.append(label)
            label = label + 1
        
        
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    
    def histogram_equalization(img_in):
    # segregate color streams
        b,g,r = cv2.split(img_in)
        h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
        h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
        h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf    
        cdf_b = np.cumsum(h_b)  
        cdf_g = np.cumsum(h_g)
        cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values 
        cdf_m_b = np.ma.masked_equal(cdf_b,0)
        cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
        cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')

        cdf_m_g = np.ma.masked_equal(cdf_g,0)
        cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
        cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
        cdf_m_r = np.ma.masked_equal(cdf_r,0)
        cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
        cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
    # merge the images in the three channels
        img_b = cdf_final_b[b]
        img_g = cdf_final_g[g]
        img_r = cdf_final_r[r]

        img_out = cv2.merge((img_b, img_g, img_r))
    # validation
        equ_b = cv2.equalizeHist(b)
        equ_g = cv2.equalizeHist(g)
        equ_r = cv2.equalizeHist(r)
        equ = cv2.merge((equ_b, equ_g, equ_r))
    
        return img_out
    
    def __getitem__(self, idx):
        
        images_path = self.root + self.images[idx]
        
        #images_path = self.images[idx]
        #print(images_path)
        labels = self.labels[idx]
        #print(labels)
        
        
        #read the file to a PIL or cv2 like file
        img = mpimg.imread(images_path)
        
        
        #insert your transferm method
        img2 = histogram_equalization(img)
        
        pil_image = Image.fromarray(img2)
        
        
        
        
        
        #transfrom image to tensor
        trans1 = transforms.ToTensor()
        tensor_image = trans1(pil_image)
        
        return tensor_image, labels


# In[15]:


# NEW CUSTOM DATA LOADER WITH PROPER SPLIT
class CustomDatasetTest(Dataset):
    def __init__(self, path, transform=None):
        
        images = []
        labels = []
        label = 0
        
        self.root = path
        
        for folder_name in os.listdir(self.root):
            if 'DS_Store' in folder_name:
                continue
            self.folder_name = folder_name + "/_test/"
            for file_name in os.listdir(self.root + self.folder_name):
                if 'DS_Store' in file_name:
                    continue
                true_file = self.folder_name + file_name
                #print(true_file)
                images.append(true_file)
                labels.append(label)
            label = label + 1
        
        
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    
    def histogram_equalization(img_in):
    # segregate color streams
        b,g,r = cv2.split(img_in)
        h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
        h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
        h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf    
        cdf_b = np.cumsum(h_b)  
        cdf_g = np.cumsum(h_g)
        cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values 
        cdf_m_b = np.ma.masked_equal(cdf_b,0)
        cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
        cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')

        cdf_m_g = np.ma.masked_equal(cdf_g,0)
        cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
        cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
        cdf_m_r = np.ma.masked_equal(cdf_r,0)
        cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
        cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
    # merge the images in the three channels
        img_b = cdf_final_b[b]
        img_g = cdf_final_g[g]
        img_r = cdf_final_r[r]

        img_out = cv2.merge((img_b, img_g, img_r))
    # validation
        equ_b = cv2.equalizeHist(b)
        equ_g = cv2.equalizeHist(g)
        equ_r = cv2.equalizeHist(r)
        equ = cv2.merge((equ_b, equ_g, equ_r))
    
        return img_out

    def adjust_gamma(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)  

    def __getitem__(self, idx):
        
        images_path = self.root + self.images[idx]
        
        #images_path = self.images[idx]
        #print(images_path)
        labels = self.labels[idx]
        #print(labels)
        
        
        #read the file to a PIL or cv2 like file
        img = mpimg.imread(images_path)
        
        
        #insert your transferm method
        img2 = adjust_gamma(histogram_equalization(img))
        
        pil_image = Image.fromarray(img2)
        
        
        
        
        
        #transfrom image to tensor
        trans1 = transforms.ToTensor()
        tensor_image = trans1(pil_image)
        
        return tensor_image, labels


# In[16]:


""" OLD CUSTOM DATA LOADER
class CustomDataset(Dataset):
    def __init__(self, path, transform=None):
        
        images = []
        labels = []
        label = 0
        
        self.root = path
        
        for folder_name in os.listdir(self.root):
            if 'DS_Store' in folder_name:
                continue
            self.folder_name = folder_name + '/'
            for file_name in os.listdir(self.root + self.folder_name):
                if 'DS_Store' in file_name:
                    continue
                true_file = self.folder_name + file_name
                #print(true_file)
                images.append(true_file)
                labels.append(label)
            label = label + 1
        
        
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    
    def histogram_equalization(img_in):
    # segregate color streams
        b,g,r = cv2.split(img_in)
        h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
        h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
        h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf    
        cdf_b = np.cumsum(h_b)  
        cdf_g = np.cumsum(h_g)
        cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values 
        cdf_m_b = np.ma.masked_equal(cdf_b,0)
        cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
        cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')

        cdf_m_g = np.ma.masked_equal(cdf_g,0)
        cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
        cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
        cdf_m_r = np.ma.masked_equal(cdf_r,0)
        cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
        cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
    # merge the images in the three channels
        img_b = cdf_final_b[b]
        img_g = cdf_final_g[g]
        img_r = cdf_final_r[r]

        img_out = cv2.merge((img_b, img_g, img_r))
    # validation
        equ_b = cv2.equalizeHist(b)
        equ_g = cv2.equalizeHist(g)
        equ_r = cv2.equalizeHist(r)
        equ = cv2.merge((equ_b, equ_g, equ_r))
    
        return img_out
    
    def __getitem__(self, idx):
        
        images_path = self.root + self.images[idx]
        
        #images_path = self.images[idx]
        #print(images_path)
        labels = self.labels[idx]
        #print(labels)
        
        
        #read the file to a PIL or cv2 like file
        img = mpimg.imread(images_path)
        
        
        #insert your transferm method
        img2 = histogram_equalization(img)
        
        pil_image = Image.fromarray(img2)
        
        
        
        
        
        #transfrom image to tensor
        trans1 = transforms.ToTensor()
        tensor_image = trans1(pil_image)
        
        return tensor_image, labels
"""


# In[17]:


path = 'Frames_PropSplit/'


# In[18]:


train_dataset = CustomDatasetTrain(path)


# In[19]:


test_dataset = CustomDatasetTest(path)


# In[19]:


data_loader = DataLoader(dataset, batch_size=16,
        num_workers=0,
        shuffle=True)


# In[20]:


def histogram_equalization(img_in):
# segregate color streams
    b,g,r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
# calculate cdf    
    cdf_b = np.cumsum(h_b)  
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)
    
# mask all pixels with value=0 and replace it with mean of the pixel values 
    cdf_m_b = np.ma.masked_equal(cdf_b,0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')
  
    cdf_m_g = np.ma.masked_equal(cdf_g,0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
    cdf_m_r = np.ma.masked_equal(cdf_r,0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
# merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]
  
    img_out = cv2.merge((img_b, img_g, img_r))
# validation
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    equ = cv2.merge((equ_b, equ_g, equ_r))
    #print(equ)
    #cv2.imwrite('output_name.png', equ)
    return img_out


# In[18]:







import os
label = 0
for dirName, subdirList, fileList in os.walk(path):
    if ('DS_Store' in dirName):
        continue
    print(dirName)
    cnt = 0
    print(label)
    for fname in fileList:
        if ('DS_Store' in fname):
            continue
        if (cnt == 0):
            cnt = cnt+1
            fname = os.path.join(dirName, fname)
            print(fname)
            img = mpimg.imread(fname)
            print(type(img))
            print(img.shape)
            imgplot = plt.imshow(img)
            plt.show()
            
            img2 = histogram_equalization(img)
            imgplot = plt.imshow(img2)
            plt.show()
    label = label + 1

"""
import glob
from PIL import Image
from torchvision import transforms
images=glob.glob("Frames/_Push_Frames/Push_7_10_58.jpg")
for image in images:
    img = Image.open(image)
    img2 = histogram_equalization(img)
    trans = transforms.ToPILImage()
    trans1 = transforms.ToTensor()
    plt.imshow(trans(trans1(img)))
"""


# In[21]:




train_set, val_set = torch.utils.data.random_split(dataset, [70000, 15019])


# In[21]:


train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle =True, num_workers =2)


# In[22]:


#val dataloader
val_dataloader = DataLoader(test_dataset, batch_size=16, shuffle =True, num_workers =2)


# In[23]:


print(type(val_dataloader))


# In[24]:


print(len(test_dataset))


# In[25]:


from tqdm.notebook import tqdm


# In[26]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# In[ ]:


for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for batch in tqdm(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print(running_loss)
    torch.save(model.state_dict(), f'model-{epoch}.pth')
    
    # validation loop code
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.to(device)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #correct = correct.to(device)

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

print('Finished Training')


# In[ ]:


torch.save(model.state_dict(), 'model.pth')


# In[ ]:


from tqdm import trange

iterations = 20


# In[ ]:


# how to load a saved model


# In[ ]:


"""# validation loop code
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in val_dataloader:
        images, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))"""


# In[1]:


`pwd


# In[ ]:




