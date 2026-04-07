"""Mostly modified from ros/test_images_segmentation.py"""

import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import networks
from fcn.config import cfg, cfg_from_file
from fcn.test_dataset import test_sample
from utils.mask import visualize_segmentation

from polygrasp.segmentation_rpc import SegmentationServer


def get_path(path):
    if os.path.isabs(path):
        return path
    root_dir = os.path.join(networks.__path__[0], "..", "..")
    return os.path.join(root_dir, path)



def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img



def parse_args():
    """
    Parse input arguments
    """
    import argparse

    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--instance', dest='instance_id', help='PoseCNN instance id to use',
                        default=0, type=int)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

    # Added
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default="data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth", type=str)
    parser.add_argument('--pretrained_crop', dest='pretrained_crop',
                        help='initialize with pretrained checkpoint for crops',
                        default="data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth", type=str)
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default="seg_resnet34_8s_embedding", type=str)
    parser.add_argument('--num-classes', help='number of classes', default=2, type=int)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    print(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = args.gpu_id
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    cfg.instance_id = args.instance_id
    num_classes = args.num_classes
    cfg.MODE = 'TEST'
    cfg.TEST.VISUALIZE = False
    print('GPU device {:d}'.format(args.gpu_id))

    num_classes = args.num_classes

    network_data = torch.load(get_path(args.pretrained))

    network = networks.__dict__[args.network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[0]).cuda(device=cfg.device)
    cudnn.benchmark = True
    network.eval()

    network_data_crop = torch.load(get_path(args.pretrained_crop))
    network_crop = networks.__dict__[args.network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda(device=cfg.device)
    network_crop = torch.nn.DataParallel(network_crop, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
    network_crop.eval()

    def run_network(im_color, depth_img, fx, fy, px, py):
        im = im_color.astype(np.float32)
        im_tensor = torch.from_numpy(im) / 255.0
        pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        im_tensor -= pixel_mean
        image_blob = im_tensor.permute(2, 0, 1)
        sample = {'image_color': image_blob.unsqueeze(0)}

        height = im_color.shape[0]
        width = im_color.shape[1]
        xyz_img = compute_xyz(depth_img, fx, fy, px, py, height, width)
        depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
        sample['depth'] = depth_blob.unsqueeze(0)

        out_label, out_label_refined = test_sample(sample, network, network_crop, num_seeds=50)

        # publish segmentation mask
        label = out_label[0].cpu().numpy()

        num_object = len(np.unique(label)) - 1
        print('%d objects' % (num_object))

        label_refined = None
        if out_label_refined is not None:
            label_refined = out_label_refined[0].cpu().numpy()

        # publish segmentation images
        im_label_refined = None
        im_label = visualize_segmentation(im_color[:, :, (2, 1, 0)], label, return_rgb=True)
        if out_label_refined is not None:
            im_label_refined = visualize_segmentation(im_color[:, :, (2, 1, 0)], label_refined, return_rgb=True)
        
        return label, label_refined, im_label, im_label_refined
    
    # import cv2
    # rgb = cv2.imread("/home/yixinlin/dev/UnseenObjectClustering/data/demo/000000-color.png")
    # d = (cv2.imread("/home/yixinlin/dev/UnseenObjectClustering/data/demo/000000-depth.png", cv2.IMREAD_ANYDEPTH) / 1000.).astype(np.float32)

    # rgbd = np.load("/home/yixinlin/dev/farsighted-mpc/third_party/fairo/perception/sandbox/polygrasp/test.npy")
    # rgb = rgbd[:, :, :3].astype(np.uint8)
    # d = (rgbd[:, :, 3] / 1000.).astype(np.float32)
    # import time

    # prev_time = time.time()
    # with torch.no_grad():
    #     label, label_refined, im_label, im_label_refined = run_network(rgb, d, fx=617.8477783203125, fy=618.071044921875, px=331.7496032714844, py=248.904541015625)
    
    # print(f"Took {time.time() - prev_time}s")
            


    class RgbdServer(SegmentationServer):
        def _get_segmentations(self, rgbd):
            rgb = rgbd[:, :, :3].astype(np.uint8)
            d = (rgbd[:, :, 3] / 1000.).astype(np.float32)

            print("Running segmentation network...")
            prev_time = time.time()

            with torch.no_grad():
                label, label_refined, im_label, im_label_refined = run_network(rgb, d, fx=617.8477783203125, fy=618.071044921875, px=331.7496032714844, py=248.904541015625)
            
            print(f"Done. Took {time.time() - prev_time}s")
            return label
                # cv2.imshow("test", im_label)

                # import matplotlib.pyplot as plt
                # plt.imshow(im_label_refined)
                # plt.show()

                # import pdb; pdb.set_trace()
    server = RgbdServer()
    server.start()
