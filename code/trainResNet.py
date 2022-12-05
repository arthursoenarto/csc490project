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
from ResNet import resmodel, Class_DataLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(model,
          train_data,
          batch_size=20,
          learning_rate=0.001,
          weight_decay=0.0,
          num_iter=1000, plot=False):

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size)

    criterion = nn.MultiLabelSoftMarginLoss()

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
            if n >= num_iter: 
                break
            if imgs.size()[0] < batch_size:
                continue

            # If using the GPU, we need to move the images and labels to the GPU:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            out = model(imgs.float()) # Forward pass. Also can call model(imgs)
            labels = labels.float().unsqueeze(1).to(device=DEVICE)
            loss = criterion(out, labels) # Compute the loss
            loss.backward() # Compute the gradients 
            optimizer.step() # Like the update() 
            optimizer.zero_grad() # Like the cleaup() 

            # Save the current training information
            if plot:
                iters.append(n)
                losses.append(float(loss))             # compute loss

            n += 1
            print("iter: " + str(n))

    if plot:
        plt.title("Learning Curve")
        plt.plot(iters, losses, label="Train")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig('./code/saved_models/learningcurveforRESNETCLASSIFICATIONBATCH100IT100.png')
        plt.close()

        

    return 1

if __name__ == "__main__":
    r = A.Compose(
        [
            A.Resize(height=90, width=90),
            ToTensorV2(),
        ],
    )
    
    training = Class_DataLoader("./data/ISIC2018_Task3_Training_Input", "./data/ISIC2018_Task3_Training_GroundTruth.csv", r)
    model = resmodel()
    train(model, training, batch_size=16, learning_rate=0.001, weight_decay=0.0, num_iter=100, plot=True)
    torch.save(model.state_dict(), "./code/saved_models/classResNet_modelBATCH100It100.pth")
