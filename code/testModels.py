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
    # data = Load_All_Segmentation_DataLoader("./data/ISIC2018_Task1-2_Validation_Input", "./data/ISIC2018_Task1_Validation_GroundTruth", "./data/ISIC2018_Task2_Validation_GroundTruth", transformation=transform)

    data = Segmentation_DataLoader("./data/ISIC2018_Task1-2_Validation_Input", "./data/ISIC2018_Task1_Validation_GroundTruth", transformation=transform)
    data1 = Globules_Segmentation_DataLoader("./data/ISIC2018_Task1-2_Validation_Input", "./data/ISIC2018_Task2_Validation_GroundTruth", transformation=transform)
    data2 = Milia_Like_Cyst_Segmentation_DataLoader("./data/ISIC2018_Task1-2_Validation_Input", "./data/ISIC2018_Task2_Validation_GroundTruth", transformation=transform)
    data3 = Negative_Network_Segmentation_DataLoader("./data/ISIC2018_Task1-2_Validation_Input", "./data/ISIC2018_Task2_Validation_GroundTruth", transformation=transform)
    data4 = Pigment_Network_Segmentation_DataLoader("./data/ISIC2018_Task1-2_Validation_Input", "./data/ISIC2018_Task2_Validation_GroundTruth", transformation=transform)
    data5 = Streaks_Segmentation_DataLoader("./data/ISIC2018_Task1-2_Validation_Input", "./data/ISIC2018_Task2_Validation_GroundTruth", transformation=transform)

    train_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    train_loader1 = torch.utils.data.DataLoader(data1, batch_size=1, shuffle=False)
    train_loader2 = torch.utils.data.DataLoader(data2, batch_size=1, shuffle=False)
    train_loader3 = torch.utils.data.DataLoader(data3, batch_size=1, shuffle=False)
    train_loader4 = torch.utils.data.DataLoader(data4, batch_size=1, shuffle=False)
    train_loader5 = torch.utils.data.DataLoader(data5, batch_size=1, shuffle=False)


    #load trained algorithm
    model_segmentation = model()
    model_globules = model()
    model_milia_like_cyst = model()
    model_negative_network = model()
    model_pigment_network = model()
    model_streaks = model()

    model_segmentation.load_state_dict(torch.load("./code/saved_models/seg_model16x2500.pth"))
    model_globules.load_state_dict(torch.load("./code/saved_models/globules_seg_model32x100.pth"))
    model_milia_like_cyst.load_state_dict(torch.load("./code/saved_models/milia_like_cyst_seg_model32x100.pth"))
    model_negative_network.load_state_dict(torch.load("./code/saved_models/negative_network_seg_model32x100.pth"))
    model_pigment_network.load_state_dict(torch.load("./code/saved_models/pigment_network_seg_model32x100.pth"))
    model_streaks.load_state_dict(torch.load("./code/saved_models/streaks_seg_model32x100.pth"))


    Images = []

    for imgs, labels in iter(train_loader):
        Images.append([imgs, labels])

    for loader in [train_loader1, train_loader2, train_loader3, train_loader4, train_loader5]:
        n = 0
        for imgs, labels in iter(loader):
            Images[n].append(labels)
            n += 1



    for lst in Images:
        imgs = lst[0]
        labels = lst[1]
        globule_labels = lst[2]
        milia_like_cyst_labels = lst[3]
        negative_network_labels = lst[4]
        pigment_network_labels = lst[5]
        streaks_labels = lst[6]

        fig, axarr = plt.subplots(nrows=4,ncols=4)

        # ROW 1
        plt.sca(axarr[0][0]) 
        plt.imshow(imgs.reshape(3, 90, 90).permute(1, 2, 0))
        plt.title("Image")

        plt.sca(axarr[0][1])
        plt.imshow(imgs.reshape(3, 90, 90).permute(1, 2, 0))
        prediction = torch.sigmoid(model_segmentation.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        plt.contour(pred, cmap='magma', alpha=0.5)
        prediction = torch.sigmoid(model_globules.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        plt.contour(pred, cmap='jet', alpha=0.5)
        prediction = torch.sigmoid(model_milia_like_cyst.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        plt.contour(pred, cmap='jet', alpha=0.5)
        prediction = torch.sigmoid(model_negative_network.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        plt.contour(pred, cmap='jet', alpha=0.5)
        prediction = torch.sigmoid(model_pigment_network.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        plt.contour(pred, cmap='jet', alpha=0.5)
        prediction = torch.sigmoid(model_pigment_network.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        plt.contour(pred, cmap='jet', alpha=0.5)
        prediction = torch.sigmoid(model_streaks.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        plt.contour(pred, cmap='jet', alpha=0.5)       

        plt.sca(axarr[0][2])
        plt.imshow(imgs.reshape(3, 90, 90).permute(1, 2, 0))
        plt.contour(np.squeeze(labels), cmap='magma', alpha=0.5)
        plt.contour(np.squeeze(torch.flip(globule_labels, dims=(0,))), cmap='jet', alpha=0.5)
        plt.contour(np.squeeze(torch.flip(milia_like_cyst_labels, dims=(0,))), cmap='jet', alpha=0.5)
        plt.contour(np.squeeze(torch.flip(negative_network_labels, dims=(0,))), cmap='jet', alpha=0.5)
        plt.contour(np.squeeze(torch.flip(pigment_network_labels, dims=(0,))), cmap='jet', alpha=0.5)
        plt.contour(np.squeeze(torch.flip(streaks_labels, dims=(0,))), cmap='jet', alpha=0.5)

        plt.sca(axarr[0][3]) 

        # ROW 2
        plt.sca(axarr[1][0])
        prediction = torch.sigmoid(model_segmentation.forward(imgs)) # use tanh instead of sigmoid?
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        plt.imshow(pred, 'gray')
        plt.title("Seg pred")

        plt.sca(axarr[1][1])
        plt.imshow(labels.permute(1, 2, 0), 'gray')
        plt.title("Seg G.T.")

        plt.sca(axarr[1][2])
        prediction = torch.sigmoid(model_globules.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        plt.imshow(pred, 'gray')
        plt.title("Globule Pred")
        
        plt.sca(axarr[1][3])
        plt.imshow(globule_labels.permute(1, 2, 0), 'gray')
        plt.title("Globule G.T.")  

        # ROW 3
        plt.sca(axarr[2][0])
        prediction = torch.sigmoid(model_milia_like_cyst.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        plt.imshow(pred, 'gray')
        plt.title("Cyst Pred")   

        plt.sca(axarr[2][1])
        plt.imshow(milia_like_cyst_labels.permute(1, 2, 0), 'gray')
        plt.title("Milia G.T.") 

        plt.sca(axarr[2][2])
        prediction = torch.sigmoid(model_negative_network.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        plt.imshow(pred, 'gray')
        plt.title("Negative Pred")

        plt.sca(axarr[2][3])
        plt.imshow(negative_network_labels.permute(1, 2, 0), 'gray')
        plt.title("Negative G.T.") 

        # ROW 4
        plt.sca(axarr[3][0])
        prediction = torch.sigmoid(model_pigment_network.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        plt.imshow(pred, 'gray')
        plt.title("Pigment Pred")

        plt.sca(axarr[3][1])
        plt.imshow(pigment_network_labels.permute(1, 2, 0), 'gray')
        plt.title("Pigment G.T.") 

        plt.sca(axarr[3][2])
        prediction = torch.sigmoid(model_streaks.forward(imgs))
        prediction = (prediction > 0.5).float()
        pred = np.squeeze(prediction.detach().numpy())
        plt.imshow(pred, 'gray')
        plt.title("Streaks Pred")

        plt.sca(axarr[3][3])
        plt.imshow(streaks_labels.permute(1, 2, 0), 'gray')
        plt.title("Streaks G.T.") 

        plt.show()





    # for imgs, labels in iter(train_loader):
    #     fig, axarr = plt.subplots(nrows=1,ncols=4)

    #     plt.sca(axarr[0]) 
    #     plt.imshow(imgs.reshape(3, 90, 90).permute(1, 2, 0))
    #     plt.title("Image")

    #     plt.sca(axarr[1])
    #     prediction = torch.sigmoid(modelGlobules.forward(imgs))
    #     prediction = (prediction > 0.5).float()
    #     pred = np.squeeze(prediction.detach().numpy())
    #     # plt.imshow(modelA(imgs).squeeze().detach().numpy())
    #     plt.imshow(pred)
    #     plt.title("Globule Prediction")

    #     plt.sca(axarr[2])
    #     prediction = torch.sigmoid(modelB.forward(imgs)) # use tanh instead of sigmoid?
    #     prediction = (prediction > 0.5).float()
    #     pred = np.squeeze(prediction.detach().numpy())
    #     # plt.imshow(modelB(imgs).squeeze().detach().numpy())
    #     plt.imshow(pred)
    #     plt.title("Prediction 16x2500")

    #     plt.sca(axarr[3])
    #     plt.imshow(labels.permute(1, 2, 0))
    #     plt.title("Ground Truth")
    #     plt.show()