import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from UNet import model
import matplotlib.pyplot as plt

class resmodel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = models.resnet34(pretrained=False)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(num_ftrs, 7)
        )
        
    def forward(self, xb):
        return self.network(xb)

class Class_DataLoader(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_labels = pd.read_csv(labels)
        self.img_dir = img_dir
        self.images = os.listdir(img_dir)
        self.transform = transform
        self.segmodel = model()
        self.segmodel.load_state_dict(torch.load("./code/saved_models/seg_model16x2500.pth"))

        SEG_TRANSFORM = A.Compose(
        [
            A.Resize(height=90, width=90),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

        self.segdata = Segmentation_DataLoader(img_dir, SEG_TRANSFORM)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # Get the path to the image
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) + ".jpg"

        image = np.array(Image.open(img_path).convert("RGB"))
        
        #unaltered image
        plt.imshow(image)
        plt.title("original Image")
        plt.show()

        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        #manipulated image
        plt.imshow(image.permute(1, 2, 0))
        plt.title("augmented original Image")
        plt.show()

        with torch.no_grad():
          mask = torch.sigmoid(self.segmodel(self.segdata[idx].unsqueeze(0)))
          mask = (mask > 0.5).float()
        
        mask = mask.squeeze(0)
        mask = mask.repeat(3, 1, 1)

        #manipulated image
        plt.imshow(mask.permute(1, 2, 0))
        plt.title("augmented original Image")
        plt.show()


        image = image*mask

        label = torch.tensor(list(self.img_labels.iloc[0])[1:])

        return image.long(), label

class Segmentation_DataLoader(Dataset):
    
    def __init__(self, image_dir, transformation=None):
        self.image_dir = image_dir
        self.transform = transformation
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image