from UNet import model 
from Dataloaders import Segmentation_DataLoader
from Dataloaders import Globules_Segmentation_DataLoader
from Dataloaders import Milia_Like_Cyst_Segmentation_DataLoader
from Dataloaders import Negative_Network_Segmentation_DataLoader
from Dataloaders import Pigment_Network_Segmentation_DataLoader
from Dataloaders import Streaks_Segmentation_DataLoader

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
import time


# code from my CSC311 Lab
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(model,
          train_data,
          # valid_data,
          batch_size=20,
          learning_rate=0.001,
          weight_decay=0.0,
          num_iter=1000, plot=False):

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    # Set the model to use the GPU if it is available
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    # Instantiate a few Python lists to track the learning curve
    iters, losses, train_acc, val_acc = [], [], [], []

    n = 0 # tracks the number of iterations
    start = time.time()
    while n < num_iter:
        for imgs, labels in iter(train_loader): # (imgs, labels) is a minibatch
            if n >= num_iter: 
                break
            if imgs.size()[0] < batch_size:
                continue

            # If using the GPU, we need to move the images and labels to the GPU:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()

            out = model.forward(imgs) # Forward pass. Also can call model(imgs)
            labels = labels.float().unsqueeze(1).to(device=DEVICE)
            loss = criterion(out, labels) # Compute the loss
            loss.backward() # Compute the gradients (like in lab 5)
            optimizer.step() # Like the update() method in lab 5
            optimizer.zero_grad() # Like the cleaup() method in lab 5

            # Save the current training information
            if plot:
                iters.append(n)
                losses.append(float(loss)/batch_size)             # compute *average* loss

            # Increment the iteration count
            n += 1
            end = time.time()
            print("iteration: " + str(n) + " , time elapsed: " + str((end - start)) + "sec")

    if plot:
        plt.title("Learning Curve")
        plt.plot(iters, losses, label="Train")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        # save fig task 1
        # plt.savefig('./code/saved_models/learningcurveforunetseg32x100.png')

        # save fig task 2
        # plt.savefig('./code/saved_models/learningcurveforunet_globules_seg32x100.png')
        # plt.savefig('./code/saved_models/learningcurveforunet_milia_like_cyst_seg32x100.png')
        # plt.savefig('./code/saved_models/learningcurveforunet_negative_network_seg32x100.png')
        # plt.savefig('./code/saved_models/learningcurveforunet_pigment_network_seg32x100.png')
        plt.savefig('./code/saved_models/learningcurveforunet_streaks_seg32x100.png')

    return losses[-1]


if __name__ == "__main__":
    start = time.time()
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

    #load dataset task 1
    # training = Segmentation_DataLoader("./data/ISIC2018_Task1-2_Training_Input", "./data/ISIC2018_Task1_Training_GroundTruth", transformation=transform)

    # load dataset task 2
    # training = Globules_Segmentation_DataLoader("./data/ISIC2018_Task1-2_Training_Input", "./data/ISIC2018_Task2_Training_GroundTruth", transformation=transform)
    # training = Milia_Like_Cyst_Segmentation_DataLoader("./data/ISIC2018_Task1-2_Training_Input", "./data/ISIC2018_Task2_Training_GroundTruth", transformation=transform)
    # training = Negative_Network_Segmentation_DataLoader("./data/ISIC2018_Task1-2_Training_Input", "./data/ISIC2018_Task2_Training_GroundTruth", transformation=transform)
    # training = Pigment_Network_Segmentation_DataLoader("./data/ISIC2018_Task1-2_Training_Input", "./data/ISIC2018_Task2_Training_GroundTruth", transformation=transform)
    training = Streaks_Segmentation_DataLoader("./data/ISIC2018_Task1-2_Training_Input", "./data/ISIC2018_Task2_Training_GroundTruth", transformation=transform)

    model = model()
    train(model, training, batch_size=32, learning_rate=0.001, weight_decay=0.0, num_iter=100, plot=True)

    # save model task 1
    # torch.save(model.state_dict(), "./code/saved_models/seg_model32x100.pth")

    # save model task 2
    # torch.save(model.state_dict(), "./code/saved_models/globules_seg_model32x100.pth")
    # torch.save(model.state_dict(), "./code/saved_models/milia_like_cyst_seg_model32x100.pth")
    # torch.save(model.state_dict(), "./code/saved_models/negative_network_seg_model32x100.pth")
    # torch.save(model.state_dict(), "./code/saved_models/pigment_network_seg_model32x100.pth")
    torch.save(model.state_dict(), "./code/saved_models/streaks_seg_model32x100.pth")

    end = time.time()
    print(str((end - start)//60) + "MINS")
