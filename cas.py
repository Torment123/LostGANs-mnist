import argparse
import os
from ast import literal_eval

import imageio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import argmax
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets

from sklearn.model_selection import train_test_split


class MNISTDataset(datasets.MNIST):
    def dt(self):
        return None


class LoadSamples(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=0, index_col=0)
        self.root_dir = root_dir
        self.transform = transform

        self.annotations['labels'] = self.annotations['labels'].apply(literal_eval)
        self.annotations['bbox'] = self.annotations['bbox'].apply(literal_eval)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #    idx = idx.tolist()
        imgs, lbls, bbs = [], [], []
        img_name = os.path.join(self.root_dir,
                                self.annotations.loc[idx, 'name'])
        # image = imageio.imread(img_name)
        image = Image.open(img_name)
        image = image.convert('L')
        image = np.expand_dims(image, axis=-1)

        labels = self.annotations.loc[idx, 'labels']
        # labels = np.array([labels])
        bbox = self.annotations.loc[idx, 'bbox']
        # bbox = np.array([bbox])
        if self.transform is not None:
            image = self.transform(image)

        return image, labels, bbox


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def main(args):
    samples_dir = args.sample_path

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    fake_dataset = LoadSamples(csv_file=samples_dir + '/crop_annotations.csv',
                               root_dir=samples_dir + '/fake/',
                               transform=transform)

    # TRAIN ON FAKE
    fake_dataloader = DataLoader(
        fake_dataset, batch_size=32,
        drop_last=True, shuffle=True)

    real_dataset = LoadSamples(csv_file=samples_dir + '/crop_annotations.csv',
                               root_dir=samples_dir + '/real/',
                               transform=transform)
    num_images = 10000  # Num images to test on
    real_subset = torch.utils.data.Subset(real_dataset, np.random.choice(len(real_dataset), num_images, False))

    # TEST ON REAL
    real_dataloader = DataLoader(
        real_subset, batch_size=32,
        drop_last=True, shuffle=True)

    writer = SummaryWriter(os.path.join(samples_dir, 'log'))

    model = Net().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    """Train/Test on Real Images"""
    # train_set, test_set = torch.utils.data.random_split(real_dataset, [80000, 10000])
    # print(len(train_set), len(test_set))
    # train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=10, shuffle=True)

    # ######################################################################################################################
    """Load Original MNIST Dataset"""
    # mnist_dir = '/netscratch/asharma/ds/mnist'
    # train_loader = torch.utils.data.DataLoader(
    #     MNISTDataset(mnist_dir, train=True, download=False,
    #                    transform=transform),
    #     batch_size=10, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(
    #     MNISTDataset(mnist_dir, train=False, transform=transform),
    #     batch_size=10, shuffle=True)
    # ######################################################################################################################

    """SAVE BATCH AS PLOT"""
    # for batch_1 in train_loader:
    #     batch = batch_1
    #     break
    #
    # print(batch[0].shape)  # as batch[0] contains the image pixels -> tensors
    # print(batch[1])  # batch[1] contains the labels -> tensors
    #
    # plt.figure(figsize=(12, 8))
    # for i in range(batch[0].shape[0]):
    #     plt.subplot(1, 10, i + 1)
    #     plt.axis('off')
    #     plt.imshow(batch[0][i].reshape(28, 28), cmap='gray')
    #     # plt.title(str(batch[1][i]))
    #     plt.savefig('digit_mnist.png')
    # plt.show()

    # NOTE: TRAIN on FAKE. TEST on REAL.
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        # train(model, fake_dataloader, optimizer, epoch, writer)
        # test(model, real_dataloader, writer)
        for i, data in enumerate(fake_dataloader):
            real_images, label, bbox = data
            # bbox = torch.cat(bbox)
            label = torch.cat(label)
            real_images, label = real_images.float().cuda(), label.cuda()

            optimizer.zero_grad()
            output = model(real_images)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()
            # running_loss += loss.item()
            if i % 100 == 0:  # print every 100 mini-batches
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(real_images), len(fake_dataloader.dataset),
                           100. * i / len(fake_dataloader), loss.item()))
                writer.add_scalar('Training Loss', loss.item(), epoch * len(fake_dataloader) + (i + 1))
        if epoch % 1 == 0:
            torch.save(model.state_dict(),
                       os.path.join(samples_dir, 'model/', 'G_%d.pth' % (epoch + 1)))

        correct = 0
        test_loss = 0
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(real_dataloader):
                real_images, label, bbox = data
                # bbox = torch.cat(bbox)
                label = torch.cat(label)
                real_images, label = real_images.float().cuda(), label.cuda()

                output = model(real_images)
                test_loss += F.nll_loss(output, label, size_average=False).item()  # sum up batch loss
                pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
                # correct += pred.eq(label.data.view_as(pred)).long().cpu().sum()
                correct += pred.eq(label.view_as(pred)).sum().item()
                if i % 100 == 0:
                    writer.add_scalar('Test Loss', test_loss, epoch * len(real_dataloader) + (i + 1))

            test_loss /= len(real_dataloader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(real_dataloader.dataset),
                100. * correct / len(real_dataloader.dataset)))
            writer.add_scalar('Accuracy', correct / len(real_dataloader.dataset), epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, default='samples',
                        help='path to load the dataset from')
    parser.add_argument('--epochs', type=str, default=10,
                        help='Num epochs to train')
    args = parser.parse_args()
    main(args)
