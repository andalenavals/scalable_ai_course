"""
Exercise sheet 5 - Hyperparameter optimization

This file provides a possible solution for exercise sheet 5.
You are free to use your own implementation.
"""

import os
import torch
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
from filelock import FileLock

from scalable_ml.second_exercise.data_loading import ErositaData, multiplot_images
from scalable_ml.fifth_exercise.ml_models import UNet
from scalable_ml.fifth_exercise.training import train_model
import scalable_ml.model_utilities as util
from ray import tune,train  # For hyperparameter tuning in Neural Networks.
from ray.tune.schedulers import ASHAScheduler
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json

OUTPUT_PATH='../output/erosita_segmentation_UNet_finetune'
DATA_FOLDER='/users/aanavarroa/original_gitrepos/scalable_ai_course/scalable_ml/data'
EXP_NAME='train_and_validate_10epochs'
if not os.path.isdir(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)
NSAMPLES_TEST=100
chip_dim = 256



def load_data(data_folder=""):
    nr_of_tiles = 8 # number of Erosita sky tiles
    # sky tile resolution: 3240 x 3240 pixels is decomposed into images of size chip_dim x chip_dim
    raw_data = False

    # read Erosita raw data and transform it into Numpy/PyTorch format
    
    if raw_data and train:
        with FileLock(os.path.expanduser('~/.data_erosita.lock')):
            ### read Erosita data from scatch ###
            images = util.read_erosita_data('../data/ErositaImages/', chip_dimension=chip_dim, nr_of_tiles=nr_of_tiles)
            exposure_map = util.read_erosita_data('../data/ExposureMap/', chip_dimension=chip_dim, nr_of_tiles=nr_of_tiles)
            targets = util.read_erosita_data('../data/Masks/', chip_dimension=chip_dim, nr_of_tiles=nr_of_tiles)

            # normalize image
            eps = 1e-8 # avoid zero division
            images = np.divide(images, exposure_map + eps, dtype=np.float32)

            images_true = images[np.any(targets, axis=(1, 2))]
            targets_true = targets[np.any(targets, axis=(1, 2))]

            print(f'Non zero {images_true.shape[0]}/{images.shape[0]}')
            #
            # multiplot_images(images[:4], title='image 256 x 256 pixels', ncols=2, logplot=True, save_fig=False)
            # multiplot_images(targets[:4], title='target 250 x 256 pixels', ncols=2, save_fig=False)

            # add channel dimension (here: 1 color channel)
            images = torch.from_numpy(np.expand_dims(images_true, axis=1))
            targets = torch.from_numpy(targets_true).long()

            # Store to numpy file
            np.savez(f'../data/erosita_unet_{nr_of_tiles}_tiles.npz', images=images, targets=targets)
    else:
        ### directly load preprocessed Erosita data ###
        filename=os.path.join(data_folder, f"erosita_unet_{nr_of_tiles}_tiles.npz")
        assert os.path.isfile(filename)
        data = np.load(filename)
        images = torch.from_numpy(data['images'])
        targets = torch.from_numpy(data['targets']).long()


    return images, targets


def train_and_validate(config, save=False, checkpoint_dir=None, exercise=2):
    # define important training hyperparameters
    batch_size = config["bs"]
    epochs = 40

    images, targets = load_data(data_folder=DATA_FOLDER)
    images=images[NSAMPLES_TEST:, :, :]
    targets=targets[NSAMPLES_TEST:, :, :]
    dataset=ErositaData(images, targets, scaling=1)

    # Split dataset in training and in validation set
    len_set,res=divmod(len(dataset),2)
    train_set,val_set=random_split(dataset, [len_set, len_set+res] )

    # create torch dataset loaders for training
    train_loader= DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader= DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Plot a few images
    # img_grid = torchvision.utils.make_grid(train_images[0:3, :, :])
    # util.matplotlib_imshow(img_grid, one_channel=True)

    # choose hardware on which to perform training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize ml_model
    ml_model = UNet(1, 2, n_layers=config["n_layers"], in_conv_channels=config["in_conv_channels"], kernel_size=config["kernel_size"], dropout_rate=config["dr"])
    ml_model.to(device)
    
    # run summary of model
    summary(ml_model, (1, chip_dim, chip_dim))
    # run summary of model (C, H, W) = (channels, heigth, width)
    # summary(ml_model, (1, 28 * 28))

    # if training mode is enabled, model is trained and its parameters are stored for future use
    losses, accuracy = train_model(ml_model, device, train_loader, val_loader, epochs, config["lr"], config["wd"], save=save, checkpoint_dir=checkpoint_dir, exercise=exercise )


def run_tune():
    param_space = {"lr": tune.choice([1e-2, 1e-3, 1e-4]),  # learning rate
                   "wd": tune.uniform(1e-4,1e-5),#weight decay(L2-regulariz)
                   "bs": tune.choice([16, 32, 64, 128, 256]),# batch size #even number
                   "dr": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4]),# dropout factor
                   "kernel_size": tune.choice([3]),  # number of neurons
                   "n_layers": tune.randint(1, 6),  # number of hidden layers
                   "in_conv_channels": tune.choice([8,16, 32, 64]), # i_conv_channel
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
    train_and_validate(bestconfig, save=True, checkpoint_dir=path, exercise=2)

    # Evaluating model in test data
    # if training mode is disabled, simply load pretrained parameters
    images, targets = load_data(data_folder=DATA_FOLDER)
    test_images=images[:NSAMPLES_TEST, :, :]
    test_targets=targets[:NSAMPLES_TEST, :, :]

    batch_size=NSAMPLES_TEST
    dataset=ErositaData(test_images, test_targets, scaling=1)
    test_loader = DataLoader(dataset, batch_size=batch_size)
    
    #checkpoint_path = best.checkpoint
    #checkpoint_path = os.path.join(best.path, f"checkpoint.pt")
    #checkpoint_path = util.get_latest_checkpoint(best.path)

    checkpoint_path = os.path.join(path, f"checkpoint.pt")
    assert os.path.isfile(checkpoint_path)

    ninput_feutures=28*28
    noutput=10
    config=bestconfig
    ml_model = UNet(in_channels=1, out_channels=2, n_layers=config["n_layers"], in_conv_channels=config["in_conv_channels"], kernel_size=config["kernel_size"], dropout_rate=config["dr"])

    checkpoint=torch.load(checkpoint_path)
    ml_model.load_state_dict(checkpoint["model_state"])
    ml_model.eval()

    # choose hardware on which to perform the model evaluation
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    # Plot a few validation images (not seen in the training)
   
    plt.clf()
    for j, (image_batch, target_batch) in enumerate(test_loader):
        if j>0: break
        image_batch, target_batch = image_batch.to(device), target_batch.to(device)
        pred = ml_model(image_batch)
        print(image_batch.shape, pred.shape, target_batch.shape)
        util.matplotlib_imshow_erosita(image_batch.detach().cpu().reshape(batch_size, chip_dim, chip_dim).numpy(), label="img", path=OUTPUT_PATH)
        util.matplotlib_imshow_erosita(target_batch.detach().cpu().numpy(), label="target", path=OUTPUT_PATH)
        util.matplotlib_imshow_erosita( np.argmax(pred.detach().cpu().numpy(),axis=1), label="pred", path=OUTPUT_PATH, vmin=0, vmax=1)
        
if __name__ == "__main__":
    bestconfig=run_tune()
    make_plots(path=OUTPUT_PATH)
    train_best(path=OUTPUT_PATH, bestconfig=bestconfig)

    
    
