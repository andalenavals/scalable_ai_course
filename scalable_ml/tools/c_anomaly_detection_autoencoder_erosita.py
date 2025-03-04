"""
this file provides an example for how to train an autoencoder model on prepared training data.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from scalable_ml.second_exercise.data_loading import multiplot_images
from scalable_ml.third_exercise.data_loading import ErositaDataUnsupervised
from scalable_ml.third_exercise.autoencoder_models import ConvAutoEncoder
from scalable_ml.third_exercise.training import train_model
import scalable_ml.model_utilities as util
import numpy as np

import os
OUTPUT_PATH='../output/erosita_anomalydetection_autoencoder'
if not os.path.isdir(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

def main(train=True):

    # define training parameters
    batch_size = 60
    epochs = 100
    nr_of_tiles = 8 # number of Erosita sky tiles
    chip_dim = 256 # sky tile resolution: 3240 x 3240 pixels is decomposed into images of size chip_dim x chip_dim
    raw_data = False

    if raw_data and train:
        ### read Erosita data from scatch ###
        images = util.read_erosita_data('../data/ErositaImages/', chip_dimension=chip_dim, nr_of_tiles=nr_of_tiles)
        exposure_map = util.read_erosita_data('../data/ExposureMap/', chip_dimension=chip_dim, nr_of_tiles=nr_of_tiles)

        # normalize image
        eps = 1e-8 # avoid zero division
        images = np.divide(images, exposure_map + eps, dtype=np.float32)

        images_true = images[np.any(targets, axis=(1, 2))]
        print(f'Non zero {images_true.shape[0]}/{images.shape[0]}')
        #
        # multiplot_images(images[:4], title='image 256 x 256 pixels', ncols=2, logplot=True, save_fig=False)
        # multiplot_images(targets[:4], title='target 250 x 256 pixels', ncols=2, save_fig=False)

        # add channel dimension (here: 1 color channel)
        images = torch.from_numpy(np.expand_dims(images_true, axis=1))

        # Store to numpy file
        np.savez(f'../data/erosita_unet_{nr_of_tiles}_tiles.npz', images=images)
    else:
        ### directly load preprocessed Erosita data ###
        data = np.load(f'../data/erosita_unet_{nr_of_tiles}_tiles.npz')
        images = torch.from_numpy(data['images'])
    
    # ToDO: Import real eROSITA data
    #images = torch.empty((0, 25), dtype=torch.float32)

    # Normalize data and create dataset object
    max_img_brightn = torch.amax(images, dim=(0, 1, 2, 3))
    erosita_data = ErositaDataUnsupervised(images, scaling=max_img_brightn.float())

    # Split dataset in training and in validation set
    len_set,res=divmod(len(erosita_data),2)
    train_set,val_set=random_split(erosita_data, [len_set, len_set+res] )

    # create torch dataset loaders for training
    train_loader= DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader= DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # choose hardware on which to perform training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f'device {device}')
    #assert False
    #
    # # initialize ml_model
    ml_model = ConvAutoEncoder(n_channels=1)
    ml_model.to(device)

    summary(ml_model, (1, chip_dim, chip_dim))

    
    if train:
        # if training mode is enabled, model is trained and its parameters are stored for future use
        losses = train_model(ml_model, device, train_loader, val_loader, epochs)
        util.store_model(ml_model, OUTPUT_PATH, f'erosita_model_{nr_of_tiles}_tiles.pth')

        # plot loss over training
        util.plot_loss_over_training(epochs, losses, batch_size, path=OUTPUT_PATH)
    else:
        # if training mode is disabled, simply load pretrained parameters
        util.load_model(ml_model,OUTPUT_PATH, f'erosita_model_{nr_of_tiles}_tiles.pth')
        ml_model.eval()

        # visualization of the model performance on new data batch (test data or validation data)
        print("Doing visualization")
        for j, image_batch in enumerate(val_loader):
            if j>0: break
            image_batch = image_batch.to(device)
            pred = ml_model(image_batch)
            print(image_batch.shape, pred.shape)
            #print(np.min(image_batch.detach().cpu().numpy()), np.max(image_batch.detach().cpu().numpy()), np.median(image_batch.detach().cpu().numpy()), np.mean(image_batch.detach().cpu().numpy()))
            #print(np.min(pred.detach().cpu().numpy()), np.max(pred.detach().cpu().numpy()), np.median(pred.detach().cpu().numpy()),np.mean(pred.detach().cpu().numpy()))
            
            img=image_batch.detach().cpu().reshape(batch_size, chip_dim, chip_dim).numpy()
            ae_img=pred.detach().cpu().reshape(batch_size, chip_dim, chip_dim).numpy()
            diff=np.abs(img-ae_img)

            util.matplotlib_imshow_erosita(img, label="img", path=OUTPUT_PATH)
            util.matplotlib_imshow_erosita(ae_img, label="pred", path=OUTPUT_PATH)
            util.matplotlib_imshow_erosita(diff, label="diff", path=OUTPUT_PATH)


if __name__ == "__main__":
    main(train=True)
    #main(train=False)
