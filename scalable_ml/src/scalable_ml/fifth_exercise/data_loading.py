from torch.utils.data import Dataset


class FMNISTDataset(Dataset):

    def __init__(self, images, labels):
        images = images.float()/255.
        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        # Size of tensor is number_of_images x 768
        # Info: torch.view is very similar to torch.reshape
        images = images.view(-1, 28 * 28)
        self.images, self.labels = images, labels

    def __getitem__(self, ix):
        image, label = self.images[ix], self.labels[ix]
        return image, label

    def __len__(self):
        return len(self.images)

class FMNISTDatasetCONV(Dataset):
    def __init__(self, images, labels):
        images = images.float()/255.
        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        # Size of tensor is number_of_images x 768
        # Info: torch.view is very similar to torch.reshape
        shape=images.shape
        images = images.view(shape[0],1,shape[1], shape[2])
        self.images, self.labels = images, labels

    def __getitem__(self, ix):
        image, label = self.images[ix], self.labels[ix]
        return image, label

    def __len__(self):
        return len(self.images)


class ErositaDataUnsupervised(Dataset):
    def __init__(self, images, scaling=1.):
        images = images / scaling
        # mask values are zero and one
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        x = self.images[ix]
        return x

class VOCData(Dataset):
    def __init__(self, images, scaling=1.):
        images = images / scaling
        # mask values are zero and one
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        x = self.images[ix]
        return x
