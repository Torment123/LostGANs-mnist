import argparse
import datetime
import os
import subprocess
import time

import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

# from data.custom_mnist_loader import CustomMNISTDataLoader
from data.mnist_loader_v2 import MNISTDataset
from model.rcnn_discriminator import *
from data.cocostuff_loader import *
# from data.vg import *
from model.resnet_generator import *
from model.sync_batchnorm import DataParallelWithCallback
from utils.logger import setup_logger

import warnings

dataset_dir = '/ds2/MScoco/'
mnist_dir = '/netscratch/asharma/ds/mnist'

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def get_dataset(dataset, img_size):
    global data
    if dataset == "coco":
        data = CocoSceneGraphDataset(image_dir=dataset_dir + 'train2017/',
                                     instances_json=dataset_dir + '/annotations/instances_val2017.json',
                                     stuff_json=dataset_dir + '/annotations/stuff_val2017.json',
                                     stuff_only=False, image_size=(img_size, img_size), left_right_flip=False)
    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        data = MNISTDataset(root=mnist_dir,
                            transform=transform)

    return data


def get_recent_commit():
    return str(subprocess.check_output(["git", "describe", "--always"]).strip().decode())


def main(args):
    # parameters
    img_size = 128
    z_dim = 128
    lamb_obj = 1.1  # 0.5, 1, 1, 0
    lamb_img = 0.1  # 1, 1, 0.5, 1
    num_classes = 184 if args.dataset == 'coco' else 10  # 179
    num_obj = 8 if args.dataset == 'coco' else 9  # 31

    # data loader
    train_data = get_dataset(args.dataset, img_size)

    dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        drop_last=True, shuffle=True, num_workers=8)

    # Load model
    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3).cuda()
    netD = CombineDiscriminator128(num_classes=num_classes).cuda()

    parallel = True
    if parallel:
        netG = DataParallelWithCallback(netG)
        netD = nn.DataParallel(netD)

    g_lr, d_lr = args.g_lr, args.d_lr
    gen_parameters = []
    for key, value in dict(netG.named_parameters()).items():
        if value.requires_grad:
            if 'mapping' in key:
                gen_parameters += [{'params': [value], 'lr': g_lr * 0.1}]
            else:
                gen_parameters += [{'params': [value], 'lr': g_lr}]

    g_optimizer = torch.optim.Adam(gen_parameters, betas=(0, 0.999))

    dis_parameters = []
    for key, value in dict(netD.named_parameters()).items():
        if value.requires_grad:
            dis_parameters += [{'params': [value], 'lr': d_lr}]
    d_optimizer = torch.optim.Adam(dis_parameters, betas=(0, 0.999))

    # get most recent commit
    commit_obj = get_recent_commit()
    current_time = time.strftime("%H_%M_%dd_%mm", time.localtime())

    # make dirs
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.makedirs(os.path.join(args.out_path, 'model/'))
    writer = SummaryWriter(os.path.join(args.out_path, 'log'))

    logger = setup_logger("lostGAN", args.out_path, 0,
                          filename=current_time + '_log.txt')
    logger.info('Commit Tag: ' + commit_obj)
    logger.info('Time: ', time.localtime())
    logger.info('No rotations. No resizing. Ob=1.1, im=0.1. 3x3 grid.')
    logger.info(netG)
    logger.info(netD)

    # labelencoder = LabelEncoder()
    # enc = OneHotEncoder()

    total_steps = len(dataloader)
    start_time = time.time()
    for epoch in range(args.total_epoch):
        netG.train()
        netD.train()
        print('epoch ', epoch)

        for idx, data in enumerate(dataloader):
            real_images, label, bbox = data
            real_images, label, bbox = real_images.cuda(), \
                                       label.long().cuda().unsqueeze(-1), \
                                       bbox.float()  # , \
            # rotation.float()  # .unsqueeze(-1)
            # rotation = labelencoder.fit_transform(rotation.view(-1)).reshape(-1, 16)
            # rotation = torch.from_numpy(rotation).float().cuda()
            # rotation = enc.fit_transform(rotation).toarray()
            # rotation = torch.from_numpy(rotation).float().cuda()  # [bs*16, 4]
            # update D network
            netD.zero_grad()
            real_images, label = real_images.float().cuda(), label.long().cuda()
            d_out_real, d_out_robj = netD(real_images, bbox, label)
            # rotation=rotation)  # d_out_robj: [460, 1]

            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()
            # d_loss_rot = torch.nn.ReLU()(1.0 - d_out_rot).mean()

            z = torch.randn(real_images.size(0), num_obj, z_dim).cuda()

            fake_images = netG(z, bbox, y=label.squeeze(dim=-1))  # rm rot

            d_out_fake, d_out_fobj = netD(fake_images.detach(), bbox, label)  # rm rot

            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()
            # d_loss_frot = torch.nn.ReLU()(1.0 + d_out_frot).mean()

            d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + \
                     lamb_img * (d_loss_real + d_loss_fake)

            d_loss.backward()
            d_optimizer.step()

            # update G network
            if (idx % 1) == 0:
                netG.zero_grad()
                g_out_fake, g_out_obj = netD(fake_images, bbox, label)  # rm rot
                g_loss_fake = - g_out_fake.mean()
                g_loss_obj = - g_out_obj.mean()
                # g_loss_rot = - g_out_rot.mean()

                g_loss = g_loss_obj * lamb_obj + g_loss_fake * lamb_img
                g_loss.backward()
                g_optimizer.step()

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                logger.info("Time Elapsed: [{}]".format(elapsed))
                logger.info("Epoch[{}/{}], Step[{}/{}], "
                            "d_out_real: {:.4f}, d_out_fake: {:.4f}, g_out_fake: {:.4f} ".format(epoch + 1,
                                                                                                 args.total_epoch,
                                                                                                 idx + 1,
                                                                                                 total_steps,
                                                                                                 d_loss_real.item(),
                                                                                                 d_loss_fake.item(),
                                                                                                 g_loss_fake.item()))

                logger.info("d_obj_real: {:.4f}, d_obj_fake: {:.4f}, g_obj_fake: {:.4f} ".format(d_loss_robj.item(),
                                                                                                 d_loss_fobj.item(),
                                                                                                 g_loss_obj.item()))

                writer.add_image("real images", make_grid(real_images.cpu().data * 0.5 + 0.5, nrow=4),
                                                epoch * len(dataloader) + idx + 1)
                writer.add_image("fake images", make_grid(fake_images.cpu().data * 0.5 + 0.5, nrow=4),
                                                epoch * len(dataloader) + idx + 1)
                writer.add_scalar('DLoss', d_loss, epoch * len(dataloader) + idx + 1)
                writer.add_scalar('DLoss/real_images', d_loss_real, epoch * len(dataloader) + idx + 1)
                writer.add_scalar('DLoss/fake_images', d_loss_fake, epoch * len(dataloader) + idx + 1)
                writer.add_scalar('DLoss/real_objects', d_loss_robj, epoch * len(dataloader) + idx + 1)
                writer.add_scalar('DLoss/fake_objects', d_loss_fobj, epoch * len(dataloader) + idx + 1)

                writer.add_scalar('GLoss', g_loss, epoch * len(dataloader) + idx + 1)
                writer.add_scalar('GLoss/fake_images', epoch * len(dataloader) + idx + 1)
                writer.add_scalar('GLoss/fake_objects', epoch * len(dataloader) + idx + 1)

        # save model
        if (epoch + 1) % 5 == 0:
            torch.save(netG.state_dict(),
                       os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch + 1)))


if __name__ == "__main__":
    commit_obj = get_recent_commit()
    current_time = time.strftime("_%H_%M_%dd_%mm", time.localtime())
    path = commit_obj + current_time

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='training dataset')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='mini-batch size of training data. Default: 32')
    parser.add_argument('--total_epoch', type=int, default=10,
                        help='number of total training epoch')
    parser.add_argument('--d_lr', type=float, default=0.0001,
                        help='learning rate for discriminator')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for generator')
    parser.add_argument('--out_path', type=str, default='./outputs/3x3_no_rot_params/' + path,
                        help='path to output files')
    args = parser.parse_args()
    main(args)
