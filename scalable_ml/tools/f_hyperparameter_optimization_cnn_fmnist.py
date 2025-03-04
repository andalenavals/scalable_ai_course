"""
Exercise sheet 5 - Hyperparameter optimization

This file provides a possible solution for exercise sheet 5.
You are free to use your own implementation.
"""

import os
import torch
import torchvision
from torch.utils.data import DataLoader
from filelock import FileLock

from scalable_ml.first_exercise.data_loading import FMNISTDatasetCONV
from scalable_ml.fifth_exercise.ml_models import ConvolutionalNeuralNetwork
from scalable_ml.fifth_exercise.training import train_model
import scalable_ml.model_utilities as util
from ray import tune,train  # For hyperparameter tuning in Neural Networks.
from ray.tune.schedulers import ASHAScheduler
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json

OUTPUT_PATH='../output/fmnist_classification_CNN_finetune'
EXP_NAME='train_and_validate_10epochs'
if not os.path.isdir(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)
NSAMPLES_TEST=100


def load_data(data_folder=f'../data/FMNIST'):
    # Store to numpy file
    # np.savez('../data/FMNIST/fmnist_image_data.npz', train_images=train_images.numpy(),
    #          train_targets=train_targets.numpy(), val_images=val_images.numpy(), val_targets=val_targets.numpy())

    # data = np.load('../data/FMNIST/fmnist_image_data.npz')
    # train_images = torch.from_numpy(data['train_images'])
    # train_targets = torch.from_numpy(data['train_targets'])
    # val_images = torch.from_numpy(data['val_images'])
    # val_targets = torch.from_numpy(data['val_targets'])

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser('~/.data_FMNIST.lock')):
        fmnist = torchvision.datasets.FashionMNIST(data_folder, download=True, train=True)

        val_fmnist = torchvision.datasets.FashionMNIST(data_folder, download=True, train=False)

    return fmnist, val_fmnist


def train_and_validate(config, save=False, checkpoint_dir=None):
    # define important training hyperparameters
    batch_size = config["bs"]
    epochs = 10

    fmnist, val_fmnist = load_data(data_folder=f'../data/FMNIST')

    train_images = fmnist.data
    train_targets = fmnist.targets

    val_images = val_fmnist.data[NSAMPLES_TEST:, :, :]
    val_targets = val_fmnist.targets[NSAMPLES_TEST:]
    
    train_fmnist_data = FMNISTDatasetCONV(train_images, train_targets)
    val_fmnist_data = FMNISTDatasetCONV(val_images, val_targets)

    # create torch dataset loaders for training
    train_loader = DataLoader(train_fmnist_data, batch_size=batch_size)
    valid_loader = DataLoader(val_fmnist_data, batch_size=batch_size)

    # Plot a few images
    # img_grid = torchvision.utils.make_grid(train_images[0:3, :, :])
    # util.matplotlib_imshow(img_grid, one_channel=True)

    # choose hardware on which to perform training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize ml_model
    ninput_feutures=28*28
    img_dim=28
    noutput=10
    ml_model = ConvolutionalNeuralNetwork(img_dim, noutput, in_channels=1, in_conv_channels=config["icc"], n_conv_layers=config["ncl"], dropout_rate=config["dr"], batchnorm=True, kernel_size=3, padding=1, nr_of_neurons_fcnn=config["nr"], nr_of_layers_fcnn=config["ly"])
    ml_model.to(device)

    # run summary of model (C, H, W) = (channels, heigth, width)
    # summary(ml_model, (1, 28 * 28))

    # if training mode is enabled, model is trained and its parameters are stored for future use
    losses, accuracy = train_model(ml_model, device, train_loader, valid_loader, epochs, config["lr"], config["wd"], save=save, checkpoint_dir=checkpoint_dir )


def run_tune():
    param_space = {"lr": tune.choice([1e-2, 1e-3, 1e-4]),  # learning rate
                   "wd": tune.uniform(1e-4, 1e-5),  # weight decay (L2-regularization)
                   "bs": tune.choice([32, 64, 128, 256, 512]),  # batch size
                   "dr": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4]),  # dropout factor
                   "nr": tune.randint(100, 1601),  # number of neurons
                   "ly": tune.randint(1, 4),  # number of hidden layers
                   "ncl": tune.randint(1, 4),  # number of convolution layers
                   "icc": tune.randint(2, 100)  #input_conv_channels
                   }

    scheduler = ASHAScheduler()
    num_samples=100
    
    # GPU support is enabled by specifying GPU resources

    #checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True)
    checkpoint_config=None
    train_and_validate_withresources = tune.with_resources(train_and_validate, {"cpu": 0, "gpu": 2})
    tuner = tune.Tuner(train_and_validate_withresources,
                       param_space=param_space,
                       tune_config=tune.TuneConfig(metric="val_accuracy",
                                                   mode="max", num_samples=num_samples,
                                                   scheduler=scheduler),
                       run_config=train.RunConfig(storage_path=os.path.abspath(OUTPUT_PATH), name=EXP_NAME, checkpoint_config=checkpoint_config) 
                       )
    
    #tuner = tune.Tuner(train_and_validate, param_space=param_space,
    #                   tune_config=tune.TuneConfig(metric="val_accuracy", mode="max", num_samples=num_samples,
    #                                               scheduler=scheduler), checkpoint_at_end=True, checkpoint_freq=1)

    results = tuner.fit()

    best_result=results.get_best_result(mode="max", metric="val_accuracy")
    print("best configuration: ", best_result )
    return best_result.config


def make_plots(path=None):
    filename=os.path.join(os.path.abspath(OUTPUT_PATH),EXP_NAME)
    tuner = tune.Tuner.restore(filename, trainable=train_and_validate)
    results = tuner.get_results()
    best=results.get_best_result()
    #df = results.get_dataframe()
    

    ## Metrics plots all set ups sampled
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)
    for i, result in enumerate(results):
        metrics = result.metrics
        metrics_df = result.metrics_dataframe

        val_loss = metrics_df["val_loss"]
        val_accu = metrics_df["val_accuracy"]
        epochs_arr = np.arange(len(val_accu)) + 1
        

        axs[0].plot(epochs_arr, val_loss, '-o', label='sample %i'%(i))
        axs[0].set_xlabel('epochs')
        axs[0].set_ylabel('val loss')
        axs[0].legend()
        axs[0].grid('off')
        #plt.show()
     
        axs[1].plot(epochs_arr, val_accu, '-o', label='sample %i'%(i))
        axs[1].set_xlabel('epochs')
        axs[1].set_ylabel('val accuracy')
        axs[1].legend()
        axs[1].grid('off')
        
    fig.tight_layout()
    fig.savefig(os.path.join(path,"loss_accuracy.png"))
    plt.close(fig)


    # Metric plots for best set up
    bestconfig = best.config
    print("Optimal metrics found", bestconfig)
    best_metrics_df = best.metrics_dataframe
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)
    
    metrics = result.metrics
    metrics_df = result.metrics_dataframe

    train_loss  = best_metrics_df["training_loss"]
    train_accu = best_metrics_df["training_accuracy"]   
    val_loss = best_metrics_df["val_loss"]
    val_accu = best_metrics_df["val_accuracy"]
    epochs_arr = np.arange(len(val_accu)) + 1

    axs[0].plot(epochs_arr, val_loss, '-o', label='validation set')
    axs[0].plot(epochs_arr, train_loss, '-o', label='training set')
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('val loss')
    axs[0].legend()
    axs[0].grid('off')
    #plt.show()
     
    axs[1].plot(epochs_arr, val_accu, '-o', label='validation set')
    axs[1].plot(epochs_arr, train_accu, '-o', label='training set')
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('val accuracy')
    #axs[1].legend()
    axs[1].grid('off')
        
    fig.tight_layout()
    fig.savefig(os.path.join(path,"loss_accuracy_best.png"))
    plt.close(fig)

def train_best(path=None, bestconfig=None):
    assert bestconfig is not None
    with open(os.path.join(path,'bestconfig.json'), 'w') as file:
        json.dump(bestconfig,file)
        
    train_and_validate(bestconfig, save=True, checkpoint_dir=path)

    # Evaluating model in test data
    # if training mode is disabled, simply load pretrained parameters
    data_folder = f'../data/FMNIST'
    test_fmnist = torchvision.datasets.FashionMNIST(data_folder, download=True, train=False)
    test_images = test_fmnist.data[:NSAMPLES_TEST, :, :]
    test_targets = test_fmnist.targets[:NSAMPLES_TEST]

    test_fmnist_data = FMNISTDatasetCONV(test_images, test_targets)
    test_loader = DataLoader(test_fmnist_data, batch_size=NSAMPLES_TEST)
    
    #checkpoint_path = best.checkpoint
    #checkpoint_path = os.path.join(best.path, f"checkpoint.pt")
    #checkpoint_path = util.get_latest_checkpoint(best.path)

    checkpoint_path = os.path.join(path, f"checkpoint.pt")
    assert os.path.isfile(checkpoint_path)

    ninput_feutures=28*28
    img_dim=28
    noutput=10
    ml_model = ConvolutionalNeuralNetwork(img_dim, noutput, in_channels=1, in_conv_channels=bestconfig["icc"], n_conv_layers=bestconfig["ncl"], dropout_rate=1e-5, batchnorm=True, kernel_size=3, padding=1, nr_of_neurons_fcnn=bestconfig["nr"], nr_of_layers_fcnn=bestconfig["ly"])

    checkpoint=torch.load(checkpoint_path)
    ml_model.load_state_dict(checkpoint["model_state"])
    ml_model.eval()

    # choose hardware on which to perform the model evaluation
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    # Plot a few validation images (not seen in the training)
    plt.clf()
    for (image_batch, target_batch) in test_loader:
        image_batch, target_batch = image_batch.to(device), target_batch.to(device)
        pred = ml_model(image_batch)
        #print(image_batch.shape, pred.shape, target_batch.shape)
        util.matplotlib_imshow(image_batch.detach().cpu().reshape(NSAMPLES_TEST, 28, 28), pred.detach().cpu().numpy(),
                               target_batch.detach().cpu().numpy(), path=OUTPUT_PATH)
        break # one iteration of dataloader

if __name__ == "__main__":
    bestconfig=run_tune()
    make_plots(path=OUTPUT_PATH)
    train_best(path=OUTPUT_PATH, bestconfig=bestconfig)

    
    
