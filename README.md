
# CSC490 Medical Imaging Dataset Project

### Group Name: Team 8 - Skin Cancer Challengers

#### Team members:

- Arthur Alexandro Soenarto

- Gabriel El Haddad

- Xiaoning Wang

- Syed Taha Ali

### Datasets: 
- [Skin Cancer MNIST HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [2018 ISIC Challenge](https://challenge.isic-archive.com/data/#2018)

We choose the HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. It is also the dataset of the International Skin Imaging Collaboration (ISIC) 2018 challenge.

### Introduction:

This project is an application of different machine learning models that were used for attempting the 2018 ISIC Challenge. 

This challenge is broken into three separate tasks:

- Task 1: Lesion Segmentation  
- Task 2: Lesion Attribute Detection
- Task 3: Disease Classification

We decided to combine the tasks where the final goal was classification (task 3).

### Background:

Skin Cancer is one of the most common cancers in North America. The most common cause is from overexposure to ultraviolet rays from the sun. It involves the growth of abnormal cells in the outermost skin layer (called the epidermis), which can form malignant tumors if not treated early. Since it grows in the outermost layer, this property makes skin cancer easily detectable and extremely relevant to camera based machine learning applications, which is the motivation for our project.

### Challenge Workflow

Sequence diagram:

![](https://i.imgur.com/UcORHl1.png)


### Segmentation

Segmentation is the process of associating each pixel on an image with a class label,
For task 1, our target label was the image mask of the skin lesion, so we had to perform
binary segmentation.

![](https://i.imgur.com/f8iz5Cm.png)

We first implemented UNet, a convolutional network architecture for fast and precise segmentation of images. U-Net was developed for biomedical image segmentation at the Computer Science Department of the University of Freiburg. The network is a convolutional neural network.

Model: 

![](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)










