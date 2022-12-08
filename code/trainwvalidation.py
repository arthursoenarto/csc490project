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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 100

def get_accuracy(model, data):
    data = torch.utils.data.DataLoader(data)

    if torch.cuda.is_available():
        model.cuda()

    num_correct = 0
    num_pixels = 0
    
    with torch.no_grad():
        for imgs, labels in data:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda().unsqueeze(1)
                preds = torch.sigmoid(model(imgs))
                preds = (preds > 0.5).float()
                num_correct += (preds == labels).sum()
                num_pixels += torch.numel(preds)

    return str(num_correct / num_pixels)




def train(model,
          train_data,
          valid_data,
          batch_size=20,
          learning_rate=0.001,
          weight_decay=0.0,
          num_iter=1000, plot=False):

    model= nn.DataParallel(model) # multiple GPUS
    model.to(device=DEVICE)

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
            loss.backward() # Compute the gradients 
            optimizer.step() # Like the update() 
            optimizer.zero_grad() # Like the cleaup() 

            # Save the current training information
            if plot:
                iters.append(n)
                losses.append(float(loss))             # compute loss
                train_acc.append(get_accuracy(model, train_data)) # compute training accuracy
                val_acc.append(get_accuracy(model, valid_data)) # compute validation accuracy

            # Increment the iteration count
            n += 1

    if plot:
        plt.title("Learning Curve")
        plt.plot(iters, losses, label="Train")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig('/home/csc490w/csc490project/code/saved_models/learningcurveforunetseglossBATCH100It500.png')
        plt.close()

        plt.title("Learning Curve")
        plt.plot(iters, train_acc, label="Train")
        plt.plot(iters, val_acc, label="Validation")

        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.savefig('/home/csc490w/csc490project/code/saved_models/learningcurveforunetsegaccuracy1001002.png')
        plt.close()


    # with open('/home/csc490w/csc490project/code/saved_models/validationpoints100.pkl', 'wb') as f:
    #     pickle.dump(val_acc, f)
    
    # with open('/home/csc490w/csc490project/code/saved_models/trainingpoints100.pkl', 'wb') as f:
    #     pickle.dump(train_acc, f)
        

    return 1



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
    train(model, training, validation, batch_size=BATCH_SIZE, learning_rate=0.001, weight_decay=0.0, num_iter=500, plot=True)

    torch.save(model.state_dict(), "/home/csc490w/csc490project/code/saved_models/seg_modelBATCH100It500.pth")