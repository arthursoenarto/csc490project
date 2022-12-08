from torchmetrics import JaccardIndex
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
import pickle

"""
Includes accuracy functions for segmentation models
Pixel accuracy, dice_score, and iou

"""


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 100
def get_accuracy(model, data):
    data = torch.utils.data.DataLoader(data)

    if torch.cuda.is_available():
        model.cuda()

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou = 0
    with torch.no_grad():
        for imgs, labels in data:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda().unsqueeze(1)
                preds = torch.sigmoid(model(imgs))
                preds = (preds > 0.5).float()
                num_correct += (preds == labels).sum()
                num_pixels += torch.numel(preds)
                intersection = (preds * labels).sum()
                super_union = (preds + labels).sum()
                union = super_union - intersection
                dice_score += (2 * intersection) / (super_union + 1e-8)
                iou += intersection / (union + 1e-8)

    return (num_correct / num_pixels), dice_score/len(data), iou/len(data)

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
    training = Segmentation_DataLoader("/home/csc490w/csc490project/data/segmentation_dataset/training_images", "/home/csc490w/csc490project/data/segmentation_dataset/training_groundtruth", transformation=transform)
    validation = Segmentation_DataLoader("/home/csc490w/csc490project/data/segmentation_dataset/ISIC2018_Task1-2_Validation_Input", "/home/csc490w/csc490project/data/segmentation_dataset/ISIC2018_Task1_Validation_GroundTruth", transformation=transform)
    model = model()
    model.load_state_dict(torch.load("/home/csc490w/csc490project/code/saved_models/seg_modelBATCH100It500.pth"))

    f = open("/home/csc490w/csc490project/code/saved_models/accuracies.txt", "w")
    pixel_acc, dicescore, iou = get_accuracy(model, training) 
    f.write("training " + str(pixel_acc) + " " + str(dicescore) + " " + str(iou) + "\n")
    pixel_acc, dicescore, iou = get_accuracy(model, validation)
    f.write("validation " + str(pixel_acc) + " " + str(dicescore) + " " + str(iou) + "\n")
    f.close()
