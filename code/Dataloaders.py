"""
This python file conatains the dataloader for the two tasks

    Classification_DataLoader() has three parameters:
        1. image_dir: the path to the image folders
        2. labels: the path to the csv label file
        3. transform: the transformations will apply to the images
    
    Each training data is an image, the type of torch.tensor
     - The image has the dimension of 3*450*600 without any transformations
    
    Each label is a vector, the type of torch.tensor
     - The label vector has the dimension of 6*1
       Each entry represent to different type of skin lesion
       1. MEL: Melanoma
       2. NV: Melanocytic nevi
       3. BCC: Basal cell carcinoma
       4. AKIEC: Actinic Keratoses (Solar Keratoses) and Intraepithelial Carcinoma (Bowen's disease)
       5. BKL: Benign keratosis
       6. DF: Dermatofibroma
       7. VASC: Vascular skin lesions


    Segmentation_DataLoader() has three parameters:
        1. image_dir: the path to the image folders
        2. mask_dir: the path to the ground truth folders
        3. transform: the transformations will apply to the images
    
    Each training data is an image, the type of torch.tensor
     - The image has the dimension of 3*1129*1504 without any transformations
    
    Each label (ground truth) is an image, the type of torch.tensor
     - The label has the dimension of 1*1129*1504 without any transformations

"""
import os

#for classification
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

#for segmentation
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class Classification_DataLoader(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_labels = pd.read_csv(labels)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        # Get the path to the image
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) + ".jpg"
        print(img_path)

        # Read the image into the torch.tensor() format
        image = read_image(img_path)

        # Convert the label to the tensor
        # First convert to the list then to the torch.tensor()
        label = torch.tensor(list(self.img_labels.iloc[0])[1:])

        # Check if transformation need to apply
        if self.transform:
            image = self.transform(image)
        return image, label


class Segmentation_DataLoader(Dataset):
    
    def __init__(self, image_dir, mask_dir, transformation=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transformation
        self.images = os.listdir(image_dir)
        
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_segmentation.png"))
        # image = read_image(img_path)
        # mask = read_image(mask_path)
        image = np.array(Image.open(img_path).convert("RGB")) #input images are rgb
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #output masks are grayscale
        mask[mask == 255.0] = 1.0 # one hot vector

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

class Globules_Segmentation_DataLoader(Dataset):
    
    def __init__(self, image_dir, mask_dir, transformation=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transformation
        self.images = os.listdir(image_dir)
        
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_attribute_globules.png"))
        # image = read_image(img_path)
        # mask = read_image(mask_path)
        image = np.array(Image.open(img_path).convert("RGB")) #input images are rgb
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #output masks are grayscale
        mask[mask == 255.0] = 1.0 # one hot vector

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

class Milia_Like_Cyst_Segmentation_DataLoader(Dataset):
    
    def __init__(self, image_dir, mask_dir, transformation=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transformation
        self.images = os.listdir(image_dir)
        
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_attribute_milia_like_cyst.png"))
        # image = read_image(img_path)
        # mask = read_image(mask_path)
        image = np.array(Image.open(img_path).convert("RGB")) #input images are rgb
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #output masks are grayscale
        mask[mask == 255.0] = 1.0 # one hot vector

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

class Negative_Network_Segmentation_DataLoader(Dataset):
    
    def __init__(self, image_dir, mask_dir, transformation=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transformation
        self.images = os.listdir(image_dir)
        
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_attribute_negative_network.png"))
        # image = read_image(img_path)
        # mask = read_image(mask_path)
        image = np.array(Image.open(img_path).convert("RGB")) #input images are rgb
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #output masks are grayscale
        mask[mask == 255.0] = 1.0 # one hot vector

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

class Pigment_Network_Segmentation_DataLoader(Dataset):
    
    def __init__(self, image_dir, mask_dir, transformation=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transformation
        self.images = os.listdir(image_dir)
        
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_attribute_pigment_network.png"))
        # image = read_image(img_path)
        # mask = read_image(mask_path)
        image = np.array(Image.open(img_path).convert("RGB")) #input images are rgb
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #output masks are grayscale
        mask[mask == 255.0] = 1.0 # one hot vector

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

class Streaks_Segmentation_DataLoader(Dataset):
    
    def __init__(self, image_dir, mask_dir, transformation=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transformation
        self.images = os.listdir(image_dir)
        
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_attribute_streaks.png"))
        # image = read_image(img_path)
        # mask = read_image(mask_path)
        image = np.array(Image.open(img_path).convert("RGB")) #input images are rgb
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #output masks are grayscale
        mask[mask == 255.0] = 1.0 # one hot vector

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask