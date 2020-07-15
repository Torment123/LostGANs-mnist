import random
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection


def make_bbox(x, y, w, h, grid_size):
    bbox = []
    for x, y, w, h in zip(x, y, w, h):
        bbox.append(np.array([float(x), float(y), float(w), float(h)]))
    bbox = np.vstack([bbox])
#     print(bbox)
    return bbox


class MNISTDataset(torchvision.datasets.MNIST):
    """:returns RGB Image """
    def __getitem__(self, index):
        images = []
        targets = []
        bboxes = []
        rotations = []
        x_width, y_height = [], []
        x_coord, y_coord = [], []
        canvas = Image.new('L', (128, 128), 0)
        for i in range(random.randint(3, 5)):  # adjust according to grid size
            index = random.randint(0, self.__len__() - 1)
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img.numpy(), mode='L')
            x, y = random.randint(28, 28), random.randint(28, 28)
            img = img.resize((x, y), Image.ANTIALIAS)
            new_im = Image.new('L', (x, y), 0)
            new_im.paste(img)
            x_width.append(x)
            y_height.append(y)

            images.append(new_im)
            if self.target_transform is not None:
                target = self.target_transform(target)
            targets.append(target)
        grid_size = len(images)

        pos = [28, 28]
        for im in images:
            # Get random x,y coords ot place img
            x = random.randint(10, 100)
            if not (x <= pos[0] - (pos[0] / 1) or x >= pos[0] + (pos[0] / 1)):
                x = random.randint(10, 100)
            x_coord.append(x)
            y = random.randint(10, 100)
            if not (y <= pos[1] - (pos[0] / 1) or y >= pos[1] + (pos[0] / 1)):
                y = random.randint(10, 100)
            y_coord.append(y)
            pos = [x, y]
            canvas.paste(im, (x, y))
        #         plt.imshow(canvas)

        if self.transform is not None:
            canvas = self.transform(canvas)
            canvas = torchvision.utils.make_grid(canvas, nrow=1)

        bboxes = make_bbox(x_coord, y_coord, x_width, y_height, grid_size=grid_size)
        for _ in range(len(bboxes), 5):  # max 8 digits on the canvas
            targets = np.hstack((targets, [0]))
            bboxes = np.vstack((bboxes, np.array([-0.6, -0.6, 0.5, 0.5])))

        targets = torch.tensor(np.array(targets), dtype=torch.int8)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes = torch.div(bboxes, torch.Tensor([128, 128, 128, 128]))
        return canvas, targets, bboxes
