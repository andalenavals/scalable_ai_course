"""
This script provides several utility functions and classes used for preparing and output training.

. codeauthor:: Wadim Koslow and Alexander Ruettgers

Following functionality is included

* get device for training
* storing and loading pytorch model weights
* print model parameters
* resizing of deep feature extraction
"""
import os, re
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from .second_exercise.data_loading import ChipCreator
from torchmetrics.classification import Dice
import astropy.visualization
from matplotlib.colors import ListedColormap


def store_model(model, path, model_name):
    """
    Stores parameters of pytorch model in a pth-file. If the given path already exists, parameters will be overwritten

    :param model: pytorch model
    :type model: pytorch class
    :param path: data path, where parameters will be stored
    :type path: string
    :param model_name: name of the model
    :type model_name: string
    """
    model_path = os.path.join(path, model_name)
    if os.path.exists(model_path):
        print(f'Model with name {model_name} already exists. overwriting previous model')
    else:
        print(f'Storing parameters of model {model_name}.')

    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), model_path)


def load_model(model, path, model_name):
    """
    Loads pytorch model from file that was written from function "store_model".

    :param model: pytorch model whose parameters will be overwritten with the loaded data
    :type model: pytorch class
    :param path: data path, from where parameters will be loaded
    :type path: string
    :param model_name: model name
    :type model_name: string
    """
    model.load_state_dict(torch.load(os.path.join(path, model_name), weights_only=True))


def read_erosita_data(folder, chip_dimension=256, nr_of_tiles=4):
    """
    Loads Erosita images and divides them into 12 x 12 sub-images with resolution 256 x 256 pixels

    :param folder: data path, from where images will be loaded
    :type folder: string
    :param chip_dimension:  resolution of the subimages
    :type chip_dimension: integer
    :param nr_of_tiles:  number of considered sky tiles
    :type nr_of_tiles: integer
    """
    chips_256 = ChipCreator(chip_dimension)

    data = []

    path = f'{folder}'
    ir = sorted(os.listdir(path))
    for i in ir[-nr_of_tiles:]:
        print(f'{i}')
        images = chips_256.create_chips(f'{path}/{i}')
        data.append(images)

    return np.concatenate(data, axis=0)


# @torch.no_grad()
def accuracy(x, y, model):
    model.eval() # <- let's wait till we get to dropout section
    # get the prediction matrix for a tensor of `x` images
    prediction = model(x)
    # compute if the location of maximum in each row coincides
    # with ground truth
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return 1. / (is_correct.size(dim=0)) * is_correct.cpu().numpy().sum().astype(float)


def segmentation_accuracy(x, y, model):

    prediction = model(x)
    # pred (N,2, 256,256)
    # y (N,256,256)
    prediction = model(x)
    values, indeces = prediction.max(1)
    is_correct = indeces == y
    return 1. / (is_correct.size(dim=0)) * is_correct.cpu().numpy().sum().astype(float)


def segmentation_dice(x, y, model, device):

    prediction = model(x)
    prediction = model(x)
    # Your code ...
    # result = ...
    # return result.detach().cpu().numpy()


def plot_loss_accuracy_over_training(epochs, losses, accuracy, batch_size, perf_metric='accuracy', path=None):
    epochs_arr = np.arange(epochs) + 1
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)
    

    axs[0].plot(epochs_arr, losses['train_loss'], 'bo', label='Training loss')
    axs[0].plot(epochs_arr, losses['val_loss'], 'r', label='Validation loss')
    axs[0].set_title(f'Training and validation loss when batch size is {batch_size}')
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].legend()
    axs[0].grid('off')
    #plt.show()

    axs[1].plot(epochs_arr, accuracy['train_accuracy'], 'bo', label=f'Training {perf_metric}')
    axs[1].plot(epochs_arr, accuracy['val_accuracy'], 'r', label=f'Validation {perf_metric}')

    axs[1].set_title(f'Training and validation {perf_metric} when batch size is {batch_size}')
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel(f'{perf_metric}')
    axs[1].legend()
    axs[1].grid('off')
    fig.tight_layout()
    fig.savefig(os.path.join(path,"loss_accuracy.png"))
    #plt.show()


def plot_loss_over_training(epochs, losses, batch_size, logplot=True, path=None):
    epochs_arr = np.arange(epochs) + 1
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_arr, losses['train_loss'], 'bo', label='Training loss')
    plt.plot(epochs_arr, losses['val_loss'], 'r', label='Validation loss')
    plt.title(f'Training and validation loss when batch size is {batch_size}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if logplot:
        plt.yscale('log')
    plt.legend()
    plt.grid('off')
    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(path,"loss.png"))


# Helper function for inline image display
def matplotlib_imshow(img, pred=None, target=None, one_channel=False, path=None):
    img = img * 1  # unnormalize
    npimg = img.numpy()
    for i in range(npimg.shape[0]):
        plt.imshow(npimg[i, :, :], cmap="Greys")
        if pred is not None:
            plt.title(f'prediction: {np.argmax(pred[i])}  target: {target[i]}')
        assert path is not None
        plt.savefig(os.path.join(path,"sampleimage%i"%(i)))
        #plt.show()
        if i > 5:
            break

def matplotlib_imshow_erosita(array, pred=None, target=None, one_channel=False, path=None, label='', vmin=None, vmax=None):
    array *= 1  # unnormalize   
    
    for i in range(array.shape[0]):
        plt.clf()
        vis=astropy.visualization.ZScaleInterval(n_samples=1000, contrast=0.15, max_reject=0.85,  min_npixels=5, krej=2.5, max_iterations=5)
        #vmin,vmax=vis.get_limits(array[i, :, :])
        ec=plt.imshow(array[i, :, :], cmap="Greys", vmin = vmin , vmax =vmax , origin = 'lower')
        plt.colorbar(ec)
        
        plt.savefig(os.path.join(path,"%s%i"%(label,i)))
        #plt.show()
        if i > 5:
            break

def matplotlib_imshow_segmentation(array, path=None, label='', vmin=None, vmax=None):
    
    for i in range(array.shape[0]):
        plt.clf()
        unique_classes = np.unique(array)
        num_classes = len(unique_classes)
        print("Image have %i unique classes"%(num_classes))
        cmap = ListedColormap(plt.cm.get_cmap('tab20', num_classes).colors[:num_classes])
        ec=plt.imshow(array[i, :, :], cmap=cmap, vmin = vmin , vmax =vmax , origin = 'lower')
        plt.colorbar(ec)
        
        plt.savefig(os.path.join(path,"%s%i"%(label,i)))
        #plt.show()
        if i > 5:
            break

def matplotlib_imshow_rgb(array, pred=None, target=None, one_channel=False, path=None, label='', vmin=None, vmax=None):
    # [batch, H,W,3]
    
    for i in range(array.shape[0]):
        plt.clf()
        vis=astropy.visualization.ZScaleInterval(n_samples=1000, contrast=0.15, max_reject=0.85,  min_npixels=5, krej=2.5, max_iterations=5)
        #vmin,vmax=vis.get_limits(array[i, :, :])
        ec=plt.imshow(array[i, :, :], cmap="Greys", vmin = vmin , vmax =vmax , origin = 'lower')
        plt.colorbar(ec)
        
        plt.savefig(os.path.join(path,"%s%i"%(label,i)))
        #plt.show()
        if i > 5:
            break

def matplotlib_imshow_kmeans(images,centroids, labels,  path=None,ext=''):
    images *= 1  # unnormalize   

    i=0
    maxplot_per_class=3
    setlab=set(labels)
    counter=dict(zip(setlab,[1]*len(setlab)))
    while counter and (i<images.shape[0]):
        c=labels[i]
        #print(c,i)
        i+=1
        
        if c not in counter: continue
        if counter[c] == maxplot_per_class:
            del counter[c]
            continue
        plt.clf()
        vis=astropy.visualization.ZScaleInterval(n_samples=1000, contrast=0.15, max_reject=0.85,  min_npixels=5, krej=2.5, max_iterations=5)
        vmin,vmax=vis.get_limits(images[i, :, :])
        #ec=plt.imshow(images[i, :, :], cmap="Greys", vmin = None , vmax =None , origin = 'lower')
        ec=plt.imshow(images[i, :, :], cmap="Greys", vmin = vmin , vmax =vmax , origin = 'lower')
        plt.colorbar(ec)
        plt.title("Cluster %i"%(c))
        plt.savefig(os.path.join(path,"%s%i%s"%("imagecluster", c, ext)))
        counter[c]+=1
        

    for i in range(len(setlab)):
        plt.clf()
        vis=astropy.visualization.ZScaleInterval(n_samples=1000, contrast=0.15, max_reject=0.85,  min_npixels=5, krej=2.5, max_iterations=5)
        vmin,vmax=vis.get_limits(centroids[i, :, :])
        #ec=plt.imshow(centroids[i, :, :], cmap="Greys", vmin = None , vmax =None , origin = 'lower')
        ec=plt.imshow(centroids[i, :, :], cmap="Greys", vmin = vmin , vmax =vmax , origin = 'lower')
        plt.colorbar(ec)
        plt.title("Centroid cluster %i"%(i))
        plt.savefig(os.path.join(path,"%s%i%s"%("centroidcluster", i, ext)))
        

def print_model_parameters(model, print_all_data=False):
    """
    Prints out the mean weight and bias values of every layer of a pytorch model

    :param model: pytorch model
    :type model: pytorch class
    :param print_all_data: if true, prints all weight values in addition to its mean.
    :type print_all_data: bool
    """
    params = model.named_parameters()
    print('Model parameters:')
    for name, param in params:
        print(f'Mean parameter {name}: {np.mean(param.data.cpu().numpy())}')
        if print_all_data:
            print(param.data)


def get_latest_checkpoint(folder_path):
    checkpoint_files = [f for f in os.listdir(folder_path) if f.startswith('checkpoint_') and f.endswith('.pt')]

    print(checkpoint_files)
    if not checkpoint_files:
        return None
    
    def extract_epoch(filename):
        match = re.search(r'checkpoint_(\d+)\.pt', filename)
        return int(match.group(1)) if match else -1
    
    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
    return os.path.join(folder_path, latest_checkpoint)
