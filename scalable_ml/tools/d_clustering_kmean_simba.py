"""
this file provides an example for how to perform clustering
"""

import torch
from scalable_ml.third_exercise.autoencoder_models import ConvAutoEncoder
from scalable_ml.third_exercise.data_loading import ErositaDataUnsupervised
#from scalable_ml.fourth_exercise.kmeans import KMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import scalable_ml.model_utilities as util
import matplotlib as mpl

import os
MODEL_PATH='../output/simba_anomalydetection_autoencoder'
OUTPUT_PATH='../output/simba_kmeans_autoencoder'
if not os.path.isdir(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

def main():

    # import data from camels-multifield-dataset
    # https://camels-multifield-dataset.readthedocs.io/en/latest/data.html#d-maps
    fmaps = '../data/SIMBA/Maps_T_SIMBA_LH_z0K00.npy'
    images = torch.from_numpy(np.load(fmaps))
    nimgs=100
    images=images[:nimgs,:,:]

    # save original dimension for plotting
    dim0 = images.shape[0]
    dim1 = images.shape[1]
    dim2 = images.shape[2]
    chip_dim=dim1

    print(dim0, dim1, dim2)

    images=images.reshape(dim0, 1, dim1, dim2)
    # Normalize data and create dataset object
    max_img_brightn = torch.amax(images, dim=(0, 1, 2, 3))
    data = ErositaDataUnsupervised(images, scaling=max_img_brightn.float())
    
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"


    # Clustering ##################
    # reshape data and compressed_data as (nr_of_images, nr_of_features), here each pixel is a feature

    nr_clusters = 6
    # K-Means clustering with full data complexity
    images = data.images.flatten(start_dim=1)
    
    #labels, centroids = KMeans(n_clusters=nr_clusters, max_iter=40, repetition_nr=6).fit(images)
    #labels = labels.detach().cpu().numpy()
    #centroids = centroids.detach().cpu().numpy()
    images=images.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=nr_clusters, max_iter=40).fit(images)
    labels, centroids = kmeans.labels_, kmeans.cluster_centers_

    images=images.reshape(dim0,dim1,dim2)
    centroids=centroids.reshape(nr_clusters,dim1,dim2)
    #print(labels)
    
    util.matplotlib_imshow_kmeans(images, centroids, labels, path=OUTPUT_PATH)

    # K-Means clustering with autoencoder
    # Optional: Dimensionality reduction ##################
    ml_model = ConvAutoEncoder(n_channels=1)
    ml_model.to(device)
    util.load_model(ml_model, MODEL_PATH, f'simba_autoencoder_model.pth')
    ml_model.eval()

   
    compressed_data = ml_model.encoder(data.images.to(device))
    shape_encoder = compressed_data.shape
    compressed_data = compressed_data.flatten(start_dim=1)
    #comp_labels, comp_centroids = KMeans(n_clusters=nr_clusters).fit(compressed_data)
    #comp_labels = comp_labels.detach().cpu().numpy()
    #comp_centroids = comp_centroids.detach().cpu().numpy()

    compressed_data = compressed_data.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=nr_clusters, max_iter=40).fit(compressed_data)
    comp_labels, comp_centroids = kmeans.labels_, kmeans.cluster_centers_


    dim0,C,W,H=shape_encoder
    comp_centroids=comp_centroids.reshape(nr_clusters,C,W,H)
    comp_centroids = torch.Tensor(comp_centroids)
    comp_centroids = ml_model.decoder(ml_model.bottleneck(comp_centroids))
    comp_centroids = comp_centroids.detach().cpu().numpy()
    comp_centroids=comp_centroids.reshape(nr_clusters,dim1,dim2)

    
    # Plot result for clustering CAMELS dataset
    # Note that centroids have to be reshaped to a Numpy tensor of size nr_clusters x dim1 x dim2
    # This means that each centroid is an image of size dim1 x dim2

    util.matplotlib_imshow_kmeans(images, comp_centroids, comp_labels, path=OUTPUT_PATH, ext='compressed')
    


if __name__ == "__main__":
    main()
