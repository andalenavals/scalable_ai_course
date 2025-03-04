"""
Helper routines to load the images and masks for Erosita dataset, some routines have been found on www.kaggle.com
"""

from torch.utils.data import Dataset
import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.dpi'] = 300  # increase plot resolution

class ErositaData(Dataset):
    def __init__(self, images, masks, scaling=1.):
        images = images / scaling
        # mask values are zero and one
        self.images, self.masks = images, masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        x, y = self.images[ix], self.masks[ix]
        return x, y


class ChipCreator:
    def __init__(self, chip_dimension=256):
        self.chip_dimension = chip_dimension

    def create_chips(self, image):
        # we crop the image and only consider a subset with resolution 3072 x 3072 pixels
        np_array = np.flip(fits.getdata(image)[:3072, :3072], axis=0)
        #np_array = np.flip(fits.getdata(image)[:3168, :3168], axis=0)
        #np_array = fits.getdata(image)

        # get number of chips per colomn
        n_rows = (np_array.shape[0] - 1) // self.chip_dimension + 1
        # get number of chips per row
        n_cols = (np_array.shape[1] - 1) // self.chip_dimension + 1
        # segment image into chips and return list of chips
        l = []

        for r in range(n_rows):
            for c in range(n_cols):
                start_r_idx = r * self.chip_dimension
                end_r_idx = start_r_idx + self.chip_dimension

                start_c_idx = c * self.chip_dimension
                end_c_idx = start_c_idx + self.chip_dimension
                chip = np_array[start_r_idx:end_r_idx, start_c_idx:end_c_idx]

                # Image needs to be of shape (desired_image_height, desired_image_width, 1)
                if chip.shape[0] != self.chip_dimension:
                    diff = self.chip_dimension - chip.shape[0]
                    # Add row of zeros, such that the image has the desired dimension
                    chip = np.vstack((chip, np.zeros((diff, chip.shape[1]))))
                if chip.shape[1] != self.chip_dimension:
                    diff = self.chip_dimension - chip.shape[1]
                    # Add column of zeros, such that the image has the desired dimension
                    chip = np.hstack((chip, np.zeros((chip.shape[0], diff))))

                l.append(chip)

        return np.array(l)


# Creates a combined image that consists of multiple sub-images and stores this image if necessary.
def multiplot_images(list_of_images, title, ncols=4, dpi=300, logplot=False, save_fig=False, exercise=2):
    mpl.rcParams['figure.dpi'] = dpi
    nrows = (len(list_of_images) - 1) // ncols + 1
    vsize = nrows * 6

    fig, axs = plt.subplots(nrows, ncols, figsize=(20, vsize))
    fig.tight_layout()
    fig.suptitle(f'{title}', fontsize=32, y=1.0)

    # create a single norm to be shared across all images
    image_norm = mpl.colors.Normalize(vmin=np.amin(list_of_images, axis=(0, 1, 2)), vmax=np.amax(list_of_images, axis=(0, 1, 2)))

    for r, ax in enumerate(axs):
        for c, row in enumerate(ax):
            # get the current index of the image
            i = r * ncols + c
            if i >= len(list_of_images):
                break
            ax[c].set_title(i)
            image = list_of_images[i]

            if logplot:
                im = ax[c].imshow(image, cmap='viridis', norm=mpl.colors.LogNorm())
            else:
                im = ax[c].imshow(image, cmap='viridis', norm=image_norm)

            # plot coloar indivually for each subplot
            # plt.colorbar(im, ax=ax[c], orientation='horizontal')

    plt.show()

    if save_fig:
        fig.savefig(f'../sheets/Sheet_{exercise}/figs/{title}.png', dpi=800)

