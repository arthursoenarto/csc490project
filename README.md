
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

The input data are dermoscopic lesion images in JPEG format. All lesion images are named using the scheme ISIC_<image_id>.jpg. The response data are binary mask images in PNG format. Mask images are named using the scheme ISIC_<image_id>_segmentation.png, where <image_id> matches the corresponding lesion image for the mask.

- 0: representing the background of the image, or areas outside the primary lesion
- 255: representing the foreground of the image, or areas inside the primary lesion

For training, the image was resized to 90x90 with various transformations applied and normalized such that (0, 255) -> (0, 1).

We first implemented UNet, a convolutional network architecture for fast and precise segmentation of images. U-Net was developed for biomedical image segmentation at the Computer Science Department of the University of Freiburg. The network is a convolutional neural network.

Model: 

![](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

Then, we explored a more advanced version of UNet, the DoubleUNet, which is a combination of two U-Net architectures stacked on top of each other.

![](https://raw.githubusercontent.com/DebeshJha/2020-CBMS-DoubleU-Net/master/img/DoubleU-Net.png)

Finally, we also tried a novel approach of a Triple UNet by combining the output of the single UNet and Double UNet.


Since the output images pixels are either 0's (if part of the background) or 1's (if part of the lesion), the two combination approaches we tested:

- Intersection of the predicted lesion regions of the two models using logical and
- Union of the prediction lesion regions using logical or

We had greater validation accuracy with the union of the two masks, so we decided to use union for the Triple UNet.

![](https://i.imgur.com/OKcfb9V.png)

#### Metrics

| Segmentation Model  | Accuracy | Dice Score | IOU |
| ------------- | ------------- |
| UNet (batch=100, epoch=500)  | 0.8691 | 0.7835 | 0.6189 |
| Double UNet (batch=64, epoch=50)  | 0.8590  | 0.7194 | 0.6020 |
| Triple UNet (union)  | 0.8712  | 0.7433 | 0.6317 |





