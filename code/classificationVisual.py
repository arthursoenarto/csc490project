from ResNet import model
import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torchvision.transforms as transforms
from Dataloaders import Classification_DataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ResNet import resmodel, Class_DataLoader

# Requirements: - have data loaded in the data folder
#               - have trained segmentation models for the general mask and trained ResNet model

if __name__ == "__main__":
    # Creating the datasets
    r = A.Compose(
        [
            A.Resize(height=90, width=90),
            ToTensorV2(),
        ],
    )
    
    training_dataset = Class_DataLoader("./data/ISIC2018_Task3_Validation_Input", "./data/ISIC2018_Task3_Validation_GroundTruth.csv", r)
    classification_model = resmodel()
    classification_model.load_state_dict(torch.load("./code/saved_models/classResNet_modelBATCH100It100.pth", map_location=torch.device('cpu')), strict=False)
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=False)

    # output segmentation input visual and classification prediction array in console
    for imgs, labels in iter(train_loader):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        prediction = torch.sigmoid(classification_model.forward(imgs.float()))
        pred = np.squeeze(prediction.cpu().detach().numpy() > 0.5)
        print(labels)
        print(pred)
        plt.imshow(imgs.reshape(3, 90, 90).permute(1, 2, 0))
        plt.title("Image")
        plt.show()



