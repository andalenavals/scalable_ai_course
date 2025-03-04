from torch.utils.data import Dataset


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

