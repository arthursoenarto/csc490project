from doubleunet import build_doubleunet
# from doubleunet2 import Double_UNet
# from doubleunet3 import DoubleUnet
# from doubleunet4 import doubleunet

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
import torchvision.models as models



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
    while n < num_iter:
        for imgs, labels in iter(train_loader): # (imgs, labels) is a minibatch
            if n % 10 == 0:
                print(n)
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
            # print(type(out))
            # print(type(labels))

            loss = criterion(out[1], labels)
            # loss = criterion(out, labels) # Compute the loss
            loss.backward() # Compute the gradients (like in lab 5)
            optimizer.step() # Like the update() method in lab 5
            optimizer.zero_grad() # Like the cleaup() method in lab 5

            # Save the current training information
            if plot:
                iters.append(n)
                # losses.append(float(loss)/batch_size) 
                losses.append(float(loss))              # compute *average* loss

            # Increment the iteration count
            n += 1

    if plot:
        curve_name = "double_unet_models/testDoubleUNetSeg_" + str(batch_size) + "_iters_" +str(num_iter) + ".png"
        plt.title("Learning Curve")
        plt.plot(iters, losses, label="Train")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(curve_name)
        # plt.savefig('saved_models/DoubleUNetSeg10_learningcurve_sample.png')

    return losses[-1]


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

    print(DEVICE)
    
    #load dataset
    training = Segmentation_DataLoader("data/segmentation_dataset/training_images", "data/segmentation_dataset/training_groundtruth", transformation=transform)

    # model = model()
    # train(model, training, batch_size=1, learning_rate=0.001, weight_decay=0.0, num_iter=10, plot=True)


    # First doubleunet
    # model = build_doubleunet()

    batch_sizes = [32,64,100]
    iter_sizes = [50,100]

    # train(model, training, batch_size=10, learning_rate=0.001, weight_decay=0.0, num_iter=10, plot=True)
    # model_name = "saved_models/testDoubleUNetSeg_" + str(10) + "_iters_" +str(100) + "model.pth"
    # torch.save(model.state_dict(), model_name)    

    for x in batch_sizes:
        for y in iter_sizes:
            print(x,y)
            model = build_doubleunet()
            train(model, training, batch_size=x, learning_rate=0.001, weight_decay=0.0, num_iter=y, plot=True)
            model_name = "double_unet_models/DoubleUNetSeg_" + str(x) + "_iters_" +str(y) + "model.pth"
            torch.save(model.state_dict(), model_name)