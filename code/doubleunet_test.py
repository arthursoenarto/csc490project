from torchmetrics import JaccardIndex
from doubleunet import build_doubleunet 
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

import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 100
def get_accuracy(model, single_unet, data):
    data = torch.utils.data.DataLoader(data)

    if torch.cuda.is_available():
        model.cuda()

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou = 0
    curr = 0
    predictions = []
    groundtruths = []
    
    with torch.no_grad():
        for imgs, labels in data:
            if curr %10 == 0 and curr!=0:
                print(curr)
                print((num_correct / num_pixels), dice_score/len(data), iou/len(data))
            
            labels = labels.unsqueeze(1)
            preds = model.forward(imgs)
            
            preds = torch.sigmoid(preds[1])
            preds = (preds > 0.5).float()

            num_correct += (preds == labels).sum()
            num_pixels += torch.numel(preds)
            # print(num_correct, num_pixels, preds)

            intersection = (preds * labels).sum()
            super_union = (preds + labels).sum()
            union = super_union - intersection
            dice_score += (2 * intersection) / (super_union + 1e-8)
            iou += intersection / (union + 1e-8)


            curr += 1

    print((num_correct / num_pixels), dice_score/len(data), iou/len(data))
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
    training = Segmentation_DataLoader("data/segmentation_dataset/training_images", "data/segmentation_dataset/training_groundtruth", transformation=transform)
    validation = Segmentation_DataLoader("data/segmentation_dataset/validation_images", "data/segmentation_dataset/validation_groundtruth", transformation=transform)

    doubleunet_models = [
        # "DoubleUNetSeg_32_iters_50model.pth",
        # "DoubleUNetSeg_32_iters_100model.pth",
        "DoubleUNetSeg_64_iters_50model.pth",
        # "DoubleUNetSeg_64_iters_100model.pth"
    ]

    unet_model_path = "seg_modelBATCH100It500.pth"
    single_unet = model()
    single_unet.load_state_dict(torch.load(unet_model_path, map_location=torch.device('cpu')))

    f = open("doubleunet_analysis.txt", "w")

    for x in doubleunet_models:
        print(x)

        model = build_doubleunet()
        # model.load_state_dict(torch.load("saved_models/"+x))
        model.load_state_dict(torch.load("double_unet_models/"+x))

        f.write(x)
        f.write("\n")
        f.write("training:\n")
        f.write(get_accuracy(model, training) + "\n")
        pixel_acc, dicescore, iou = get_accuracy(model, single_unet, training) 
        f.write("accuracy: " + str(pixel_acc) + ", dice: " + str(dicescore) + ", iou: " + str(iou) + "\n")
        f.write("validation:\n")
        f.write(get_accuracy(model, validation))
        pixel_acc, dicescore, iou = get_accuracy(model, single_unet, validation) 
        f.write("accuracy: " + str(pixel_acc) + ", dice: " + str(dicescore) + ", iou: " + str(iou) + "\n")
    
    f.close()
