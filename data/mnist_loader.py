import torchvision
from PIL import Image
import numpy as np
import torch
from matplotlib import patches
from matplotlib.collections import PatchCollection
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import random
import matplotlib.pyplot as plt


class MNISTDataset(torchvision.datasets.MNIST):
    def __getitem__(self, index):
        images = []
        targets = []
        bboxes = []
        rotations = []
        for i in range(16):  # adjust according to batch size
            index = random.randint(0, self.__len__() -1)
            img, target = self.data[index], int(self.targets[index])

            img = Image.fromarray(img.numpy(), mode='L')
            if self.transform is not None:
                # TODO: Train LostGAN simply with MNIST
                # x, y = random.randint(5, 28), random.randint(5, 28)
                # img = img.resize((x, y), Image.ANTIALIAS)
                # rotation = random.choice([0, 45, -45, 90, -90, 135, -135, 180])
                # img = img.rotate(rotation)
                # rotations.append(rotation)

                new_im = Image.new('L', (32, 32), 0)  # 32 to simulate padding and resulting in 128x128
                new_im.paste(img)  # , (int((32 - x) / 2), int((32 - y) / 2)))
                # WARNING: Bounding boxes do not correspond to the ones in Multi-MNIST Think of the digit '1',
                # a bounding box with width = height = 28 is not 'correct' See for MultiMNIST,
                # https://github.com/aakhundov/tf-attend-infer-repeat FIXME: bounding boxes need to adapted depending
                #  on the rotation bboxes.append([(32 - x) / 2 + 32 * (i), (32 - y) / 2 + 32 * (i), x,
                #  y])  # WARNING: Not 100% sure about this calculation bboxes.append(new_im.getbbox())

                bboxes = make_bbox(grid_size=4)
                img = self.transform(new_im)
            images.append(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            targets.append(target)

        img_grid = torchvision.utils.make_grid(images, nrow=4, padding=0)
        targets = torch.tensor(np.array(targets), dtype=torch.int8)
        bboxes = torch.tensor(np.array(bboxes), dtype=torch.int32)
        # bboxes = torch.tensor(bboxes, dtype=torch.int32)
        # rotations = torch.tensor(np.array(rotations))

        return img_grid, targets, bboxes  # , rotations


def make_bbox(grid_size=2):
    bbox = []
    w, h = 32, 32
    for i in range(grid_size):
        for j in range(grid_size):
            bbox.append(np.array([j * w, i * w, w-4, h-4]))
    return np.vstack(bbox)


def grid(images, mini_batch):
    len_images = len(images)  # no of tensors (len)
    # im = torch.cat([images[0][0], images[0][1], images[0][2], images[0][3]])
    # print(im.shape)


def make_grid(data, grid_size=2):
    mini_batch = grid_size * grid_size

    images, labels = data
    print(images.shape)
    print('end')
    labels = [labels[x:x + mini_batch] for x in range(0, len(labels), mini_batch)]
    # images = [images[x:x + mini_batch] for x in range(0, len(images), mini_batch)]
    # im = torch.cat([images[0][0], images[0][1], images[0][2], images[0][3]]).reshape((1, 56, 56))
    # images = torchvision.utils.make_grid(images, nrow=2)
    # im = images[1].reshape((1,56,56))
    # img = im.cpu().detach().numpy()
    # img = im.reshape([1,:,-1])
    # print(im.shape)
    # save_image(im,mnist_dir+'\custom_mnist\ctry\im.png', nrow=2)

    # print(images)
    labels = torch.stack(labels)
    bbox = make_bbox(grid_size)
    # print(images[0][0])
    # inputs=torch.cat((images[0][0], images[0][1]))
    # inputs = torch.cat(images.split(mini_batch), dim=2)
    # torch.cat(images, dim=0).\
    # images = images.reshape((2,1,160,160))
    im = []
    # im=images[0:4].reshape((1,1,160,160)).squeeze()
    # im=im.reshape((2, 1, 160, 160))
    # images = torch.stack(images).view(2,1,160,160)
    # print(im.shape)
    # save_image(images, mnist_dir + '\custom_mnist\ctry\im2.png', nrow=2)
    # print(labels)
    return images, labels, bbox


# mnist_dir = '/netscratch/asharma/ds/mnist/'  # 'D:\_Personal Files\Studium\Thesis\controlling-gans\LostGANs'

# transform = transforms.Compose([
#     #    transforms.Resize((80, 80)),
#     transforms.ToTensor()
#     #    transforms.Normalize((0.5,), (0.5,))
# ])

# train_loader = DataLoader(
#     MNISTDataset(mnist_dir + '\data', train=True, download=False, transform=transform),
#     batch_size=4, shuffle=False)
#
# for i, data in enumerate(train_loader):
#     #    save_image(images, mnist_dir + '\custom_mnist\images_1x1_test\{idx}.png'.format(idx=i))
#     # images, labels, bbox = make_grid(data)
#     # print(images.size(), labels)
#     # save_image(images,mnist_dir+'\custom_mnist\ctry', nrow=2)
#     img, targets, bboxes = data
#     # print(targets)
#     # print(bboxes[0][0][0:2])
#     # to_pil = transforms.ToPILImage(mode='RGB')
#     # image = to_pil(img[0])
#     # # figure, ax = plt.subplots(1)
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, aspect='equal')
#     # rect = []
#     # for i in range(16):
#     #     x = np.asarray(bboxes[1][i][0]).__int__()
#     #     y = np.asarray(bboxes[1][i][1]).__int__()
#     #     # print(rotations[1][i])
#     #     rect.append(patches.Rectangle((x, y), 28, 28,
#     #                                   # angle=rotations[1][i],
#     #                                   edgecolor='r',
#     #                                   alpha=1,
#     #                                   fill=False))  # stupid!
#     # ax.imshow(image)
#     # #    ax.add_patch(rect)
#     # ax.add_collection(PatchCollection(rect, fc='none', ec='red'))
#     # plt.savefig(mnist_dir + '\custom_mnist\ctry\imfig.png')
#     # save_image(img[1], mnist_dir + '\custom_mnist\ctry\im3.png')
#     break
