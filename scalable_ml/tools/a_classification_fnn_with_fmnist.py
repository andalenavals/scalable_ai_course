"""
Exercise sheet 1 - Image Segmentation

This file provides the basic program for solving exercise sheet 1.
You are free to use your own implementation.
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from torchsummary import summary

from scalable_ml.first_exercise.data_loading import FMNISTDataset
from scalable_ml.first_exercise.ml_models import FullyConnectedNeuralNetwork
from scalable_ml.training import train_model
import scalable_ml.model_utilities as util

import os
OUTPUT_PATH='../output/fmnist_classification_fullyconnectedNN'
if not os.path.isdir(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

def main(train=True):

    # define important training hyperparameters
    batch_size = 128
    epochs = 15

    # download data and define dataset class
    data_folder = f'../data/FMNIST'  # This can be any directory you want to store data to
    fmnist = torchvision.datasets.FashionMNIST(data_folder, download=True, train=True)
    train_images = fmnist.data
    train_targets = fmnist.targets
    train_fmnist_data = FMNISTDataset(train_images, train_targets)
    #print(train_images.shape) #(nimages, H, W)

    # In a similar way we also have to load the validation data and define a class
    val_fmnist = torchvision.datasets.FashionMNIST(data_folder, download=True, train=False)
    val_images = val_fmnist.data
    val_targets = val_fmnist.targets
    val_fmnist_data = FMNISTDataset(val_images, val_targets)

    # create torch dataset loaders for training
    train_loader = DataLoader(train_fmnist_data, batch_size=batch_size)
    valid_loader = DataLoader(val_fmnist_data, batch_size=batch_size)

    #print(train_loader.dataset.images.shape)  #(nimages, H*W)
    #print(train_loader.dataset.labels.shape)  #(nimages)
    
    # Plot a few images
    img_grid = torchvision.utils.make_grid(train_images[0:3, :, :])
    util.matplotlib_imshow(img_grid, one_channel=True, path=OUTPUT_PATH)

    # choose hardware on which to perform training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device {device}')

    # initialize ml_model
    ml_model = FullyConnectedNeuralNetwork()
    ml_model.to(device)

    # run summary of model (C, H*W) = (channels, heigth*width)
    summary(ml_model, (1, 28 * 28))

    if train:
        # if training mode is enabled, model is trained and its parameters are stored for future use
        losses, accuracy = train_model(ml_model, device, train_loader, valid_loader, epochs)
        util.store_model(ml_model, OUTPUT_PATH, f'fmnist_model.pth')

        # plot loss and accuracy over training
        util.plot_loss_accuracy_over_training(epochs, losses, accuracy, batch_size, path=OUTPUT_PATH)
    else: # training data is not required here
        # if training mode is disabled, simply load pretrained parameters
        util.load_model(ml_model, OUTPUT_PATH, f'fmnist_model.pth')
        ml_model.eval()

        # Plot a few validation images (not seen in the training)
        for (image_batch, target_batch) in valid_loader:
            image_batch, target_batch = image_batch.to(device), target_batch.to(device)
            pred = ml_model(image_batch)
            #print(image_batch.shape, pred.shape, target_batch.shape)
            util.matplotlib_imshow(image_batch.detach().cpu().reshape(batch_size, 28, 28), pred.detach().cpu().numpy(),
                                   target_batch.detach().cpu().numpy(), path=OUTPUT_PATH)
            break # one iteration of dataloader


if __name__ == "__main__":
    #main(train=True)
    main(train=False)

