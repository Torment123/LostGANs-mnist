import imageio
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from ast import literal_eval


class CustomMNISTDataLoader(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=0, index_col=0)
        self.root_dir = root_dir
        self.transform = transform

        self.annotations['labels'] = self.annotations['labels'].apply(literal_eval)
        self.annotations['bbox'] = self.annotations['bbox'].apply(literal_eval)

    def __len__(self):
        #print(len(self.annotations))
        return len(self.annotations)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #    idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.annotations.loc[idx, 'name'])
        image = imageio.imread(img_name)
        image = np.asarray(image)
        image = np.rollaxis(image, 2, 0)
        labels = self.annotations.loc[idx, 'labels']
        # labels = np.array([labels])

        bbox = self.annotations.loc[idx, 'bbox']
        # bbox = np.array([bbox])

        # sample = {'image': image, 'labels' : labels, 'bbox': bbox}

        #        if self.transform:
        #            sample = self.transform(sample)

        labels, bbox = np.hstack(labels), np.stack(bbox)
#        print(idx, labels, bbox)
        #print('image shape:', image.shape)
        return image, labels, bbox


# mnist_dir = '/netscratch/asharma/ds/mnist'
#
# dataset = CustomMNISTDataLoader(csv_file=mnist_dir + '/custom_mnist/annotations.csv',
#                                 root_dir=mnist_dir + '/custom_mnist/images/')
#
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=32,
#     drop_last=True, shuffle=False)
#
# for i, data in enumerate(dataloader):
#     real_images, label, bbox = data
#     real_images, label, bbox = real_images.cuda(), label.long().cuda().unsqueeze(-1), bbox.float()
#     print(i)
