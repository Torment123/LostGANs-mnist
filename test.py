import argparse
from collections import OrderedDict

import imageio
import pandas as pd
from torchvision import transforms

from data.cocostuff_loader import *
# from data.custom_mnist_loader import CustomMNISTDataLoader
from data.mnist_loader_v2 import MNISTDataset
from data.vg import *
from model.resnet_generator import *
from utils.util import *

dataset_dir = '/ds2/MScoco/'
mnist_dir = '/netscratch/asharma/ds/mnist'


def get_dataloader(dataset='coco', img_size=128):
    if dataset == 'coco':
        dataset = CocoSceneGraphDataset(image_dir=dataset_dir + 'val2017/',
                                        instances_json=dataset_dir + '/annotations/instances_val2017.json',
                                        stuff_json=dataset_dir+'/annotations/stuff_val2017.json',
                                        stuff_only=False, image_size=(img_size, img_size), left_right_flip=False)
    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = MNISTDataset(root=mnist_dir,
                               train=False,
                               transform=transform)  # def in function

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        drop_last=True, shuffle=False, num_workers=8)
    return dataloader


def crop(image, bbox):
    x, y, h, w = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    image = image[:, y:y + h, x:x + w]
    return image


def main(args):
    num_classes = 10  # 184 if args.dataset == 'coco' else 10
    num_o = 9  # 8 if args.dataset == 'coco' else 16
    z_dim = 128

    df = pd.DataFrame(columns={'name': None,
                               'label': None,
                               'bbox': None})

    dfg = pd.DataFrame(columns={'name': None,
                                'label': None,
                                'bbox': None})

    arrname, arrlabel, arrbbox = [], [], []

    dataloader = get_dataloader(args.dataset)

    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3).cuda()

    if not os.path.isfile(args.model_path):
        return
    state_dict = torch.load(args.model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.cuda()
    netG.eval()

    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
    thres = 2.0
    print(len(dataloader))
    for idx, data in enumerate(dataloader):
        real_images, label, bbox = data  # [bs, 3, 128, 128], [bs, 16], [bs, 16, 4]
        scaled_boxes = torch.mul(bbox, 128)
        real_images, label, bbox = real_images.cuda(), \
                                   label.long().unsqueeze(-1).cuda(), \
                                   bbox.float()

        z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().cuda()  # [1, 16, 128]
        z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float().cuda()  # [1, 2048]

        fake_images = netG.forward(z_obj, bbox.cuda(), z_im=z_im, y=label.squeeze(dim=-1))  # rm rot

#         z = torch.randn(real_images.size(0), num_o, z_dim).cuda()
#         fake_images = netG(z, bbox, y=label.squeeze(dim=-1))  # rm rot

        imgs = fake_images[0].cpu().detach().numpy()  # .transpose(1, 2, 0) #*0.5+0.5
        rimgs = real_images[0].cpu().detach().numpy()  # .transpose(1, 2, 0) #*0.5+0.5
        imgs = imgs.transpose(1, 2, 0) * 0.5 + 0.5  # [128,128,3]
        imgs = imgs * 255
        rimgs = rimgs.transpose(1, 2, 0) * 0.5 + 0.5  # [128,128,3]
        rimgs = rimgs * 255

        imageio.imwrite("{save_path}/{idx}.png".format(save_path=args.sample_path + '/fake_grid', idx=idx),
                        imgs.astype('uint8'))
        imageio.imwrite("{save_path}/{idx}.png".format(save_path=args.sample_path + '/real_grid', idx=idx),
                        rimgs.astype('uint8'))

        dg = {'name': '{idx}.png'.format(idx=idx),
              'labels': np.array2string(np.asarray(label.cpu()), separator=', '),
              'bbox': np.array2string(np.asarray(bbox.cpu()), separator=', ')}
        dfg = dfg.append(dg, ignore_index=True)
        print('Saving grid {}'.format(idx))

        for i in range(9):  # num objs
            imgs = crop(fake_images.squeeze().cpu().detach().numpy(), scaled_boxes[0][i].cpu().detach().numpy())
            rimgs = crop(real_images.squeeze().cpu().detach().numpy(), scaled_boxes[0][i].cpu().detach().numpy())

            imgs = imgs.transpose(1, 2, 0) * 0.5 + 0.5  # [128,128,3]
            imgs = imgs * 255
            rimgs = rimgs.transpose(1, 2, 0) * 0.5 + 0.5  # [128,128,3]
            rimgs = rimgs * 255

            # hard coded (mkdir) dirs real/ and fake/
            imageio.imwrite("{save_path}/{idx}{i}.png".format(save_path=args.sample_path + '/fake', idx=idx, i=i),
                            imgs.astype('uint8'))
            imageio.imwrite("{save_path}/{idx}{i}.png".format(save_path=args.sample_path + '/real', idx=idx, i=i),
                            rimgs.astype('uint8'))
            d = {'name': '{idx}{i}.png'.format(idx=idx, i=i),
                 'labels': np.array2string(np.asarray(label[0][i].cpu()), separator=', '),
                 'bbox': np.array2string(np.asarray(bbox[0][i].cpu()), separator=', ')}
            df = df.append(d, ignore_index=True)
            print('Saving image {}{}'.format(idx, i))

    print('Images Saved!')
    df = df[['name', 'labels', 'bbox']]
    df.to_csv(args.sample_path + '/crop_annotations.csv')
    dfg = dfg[['name', 'labels', 'bbox']]
    dfg.to_csv(args.sample_path + '/grid_annotations.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='training dataset')
    parser.add_argument('--model_path', type=str,
                        help='which epoch to load')
    parser.add_argument('--sample_path', type=str, default='samples',
                        help='path to save generated images')
    args = parser.parse_args()
    main(args)
