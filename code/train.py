"""
This python file contains train() function will train the model


"""
import torch
import torch.nn as nn
from importlib import import_module
import Plotter
import dataloader
import torch.optim as optim
import torchvision.transforms.functional as TF
from tqdm import tqdm
import time
import sys

def train(model,
          loss_function,
          evaluation_metic,
          train_data: dataloader,
          batch_size,
          learning_rate,
          weight_decay,
          epochs,
          path):
    
    # The train function will train the model in the way of given parameters

    criterion = loss_function

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    # Instantiate a few Python lists to track the learning curve
    losses = []
    accuracies = []

    for epoch in range(epochs):

        # Train
        with tqdm(total=len(train_data), desc="Epoch " + str(epoch+1)) as pbar:
            model.train()
            for i, data in enumerate(train_data, 0):
                # Update the progress bar
                pbar.update()
                # Load the current data
                imgs = data[0].to(DEVICE)
                labels = data[1].to(DEVICE)

                out = model.forward(imgs) # Forward pass. Also can call model(imgs)
                labels = labels.float().unsqueeze(1).to(device=DEVICE)
                loss = criterion(out, labels) # Compute the loss
                loss.backward() # Compute the gradients (like in lab 5)
                optimizer.step() # Like the update() method in lab 5
                optimizer.zero_grad() # Like the cleaup() method in lab 5

                # Save the current training information
                losses.append(loss)
        
        # Validation
        with tqdm(total=len(train_data), desc="Testing Epoch " + str(epoch+1)) as pbar:
            model.eval()
            for i, data in enumerate(train_data, 0):
                # Update the progress bar
                pbar.update()
                # Load the current data
                imgs = data[0].to(DEVICE)
                labels = data[1].to(DEVICE)
                labels_int = labels.type(torch.int64).to(DEVICE)
                with torch.no_grad():
                    prediction = torch.sigmoid(model.forward(imgs))
                    prediction = (prediction > 0.5).float()
                    
                    metric = evaluation_metic
                    accuracy = metric(prediction, labels_int)
                  
                    # Save the current training information
                    accuracies.append(accuracy)
    
    # Save the model
    torch.save(model.state_dict(), path)
    return losses, accuracies



if __name__ == "__main__":

    # Preparing for dataloaders
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

    # Set the batch_size = 32
    batch_size = 32

    # Set the learning_rate:
    learning_rate = 0.001
    weight_decay = 0

    # Retrieve all the arguments
    arguments = sys.argv
    
    # There are 6 arguments need to train the model
    # 1. Which dataset is going to use: C (classification) or S (segmentation)
    # 2. Which model is going to use: file name i.e. use UNet if the model is in UNet.py

    # 1. Retrieving the datasets
    if arguments[1] == "C":
        # Training data
        train_data_path = "./data/classification_dataset/training_images"
        # Training labels
        train_data_label_path = "./data/classification/training_labels_classification.csv"

        # Createing dataloaders
        train = DataLoaders.Classification_DataLoader(train_data_path, train_data_label_path)
        train_dl = torch.utils.data.DataLoader(training, batch_size=batch_size, shuffle=True)
    elif arguments[1] == "S":
        # Training data
        train_data_path = "./data/segmentation_dataset/training_images"
        # Training labels
        train_data_label_path = "./data/segmentation_dataset/training_groundtruth"

        # Createing dataloaders
        dataloader = DataLoaders.Segmentation_DataLoader(train_data_path, train_data_label_path)
        train_dl = torch.utils.data.DataLoader(training, batch_size=batch_size, shuffle=True)
    else:
        print("Wrong type of dataset: C for classification or S for segmentation")
        sys.exit()

    # 2. Retrieve the model
    model_lib = import_module(arguments[2])
    model = mode_lib.model()

    # 3. Retrieve the loss function (Not implemented)
    loss_function = nn.BCEWithLogitsLoss()

    # 4. Retrieve the metric function (Not implemented)
    metric_function = JaccardIndex(num_classes=2)

    # 5. Retrieve the number of epoch (Not implemented)
    epoch = 10

    # Check if the GPU available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if torch.cuda.is_available():
        print("Running on " + torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")


    # Train the model
    loss, accuracy = train(model.to(device), 
                           loss_function.to(device), 
                           metric_function.to(device), 
                           dataloader.to(device), 
                           learning_rate.to(device),
                           weight_decay.to(device),
                           epoch, 
                           arguments[2])
    
    print(loss)
    print(accuracy)

    

    

    