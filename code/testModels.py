from UNet import model 
from Dataloaders import Segmentation_DataLoader

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, random_split, DataLoader
import os
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

if __name__ == "__main__":
    transform = A.Compose(
        [
            A.Resize(height=90, width=90),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    #load dataset
    training = Segmentation_DataLoader("./data/ISIC2018_Task1-2_Validation_Input", "./data/ISIC2018_Task1_Validation_GroundTruth", transformation=transform)

    train_loader = torch.utils.data.DataLoader(training, batch_size=1, shuffle=False)

    #load trained algorithm
    modelA = model()
    modelB = model()

    modelA.load_state_dict(torch.load("./code/saved_models/seg_model100x100.pth"))
    modelB.load_state_dict(torch.load("./code/saved_models/seg_model16x2500.pth"))


    for imgs, labels in iter(train_loader):
        fig, axarr = plt.subplots(nrows=1,ncols=4)

        plt.sca(axarr[0]) 
        plt.imshow(imgs.reshape(3, 90, 90).permute(1, 2, 0))
        plt.title("Image")

        plt.sca(axarr[1])
        prediction = torch.sigmoid(modelA.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        # plt.imshow(modelA(imgs).squeeze().detach().numpy())
        plt.imshow(pred)
        plt.title("Prediction 100x100")

        plt.sca(axarr[2])
        prediction = torch.sigmoid(modelB.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        # plt.imshow(modelB(imgs).squeeze().detach().numpy())
        plt.imshow(pred)
        plt.title("Prediction 16x2500")

        plt.sca(axarr[3])
        plt.imshow(labels.permute(1, 2, 0))
        plt.title("Ground Truth")
        plt.show()