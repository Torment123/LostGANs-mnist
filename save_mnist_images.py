import argparse
import os
import numpy as np
import pandas as pd

import imageio
import torch
from torchvision import transforms
from torchvision.utils import save_image

from data.cocostuff_loader import CocoSceneGraphDataset
from data.mnist_loader import MNISTDataset

dataset_dir = '/netscratch/asharma/ds/coco/'
mnist_dir = '/netscratch/asharma/ds/mnist3x3'  # 'D:\_Personal Files\Studium\Thesis\controlling-gans\LostGANs\data'


def get_dataloader(dataset='mnist', img_size=128):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        dataset = MNISTDataset(root=mnist_dir, train=True, download=True, transform=transform)  # def in function

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=9,  # 4
        drop_last=True, shuffle=False, num_workers=1)
    return dataloader


def main(args):
    dataloader = get_dataloader(args.dataset)

    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    boxes = []
    w, h = 28, 28

    # for i in range(4):
    #     if i == 0:
    #         boxes.append(np.array([0, 0, w, h]))
    #     elif i == 1:
    #         boxes.append(np.array([28, 0, w, h]))
    #     elif i == 2:
    #         boxes.append(np.array([0, 28, w, h]))
    #     else:
    #         boxes.append(np.array([28, 28, w, h]))

    for i in range(8):
        if i == 0:
            boxes.append(np.array([0, 0, w, h]))
        elif i == 1:
            boxes.append(np.array([28, 0, w, h]))
        elif i == 2:
            boxes.append(np.array([56, 0, w, h]))
        elif i == 3:
            boxes.append(np.array([0, 28, w, h]))
        elif i == 4:
            boxes.append(np.array([28, 28, w, h]))
        elif i == 5:
            boxes.append(np.array([56, 28, w, h]))
        elif i == 6:
            boxes.append(np.array([0, 56, w, h]))
        elif i == 7:
            boxes.append(np.array([28, 56, w, h]))
        else:
            boxes.append(np.array([56, 56, w, h]))
    boxes = np.vstack(boxes)

    df = pd.DataFrame(columns={'name', 'labels', 'bbox'})

    for idx, data in enumerate(dataloader):
        real_images, label = data
        #print(np.array2string(np.asarray(boxes), separator=', '))

        save_image(real_images, mnist_dir+'/custom_mnist/images/{idx}.png'.format(idx=idx), nrow=2)
        d = {'name': '{idx}.png'.format(idx=idx),
             'labels': np.array2string(np.asarray(label), separator=', '),
             'bbox': np.array2string(np.asarray(boxes), separator=', ')}
        # df=pd.DataFrame.from_dict(data=d.items()) #.to_csv('mnist_annotations.csv', header=True)
        df = df.append(d, ignore_index=True)

    df = df[['name', 'labels', 'bbox']]
    df.to_csv(mnist_dir+'/custom_mnist/annotations.csv')


#        imgs = real_images[0].cpu().detach().numpy()  # .transpose(1, 2, 0) #*0.5+0.5
#        imgs = imgs.transpose(1, 2, 0) * 0.5 + 0.5  ## [128,128,3]
# print(imgs.shape)  ## [128,128,3]
#        imgs = imgs * 255
#        imageio.imwrite("{save_path}/sample_{idx}.png".format(save_path=args.sample_path, idx=idx),
#                        imgs.astype('uint8'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='training dataset')
    parser.add_argument('--model_path', type=str,
                        help='which epoch to load')
    parser.add_argument('--sample_path', type=str, default='/netscratch/asharma/ds/mnist3x3',
                        help='path to save generated images')
    args = parser.parse_args()
    main(args)
