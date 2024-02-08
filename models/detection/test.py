import argparse
import gc
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import DataParallel
from generate_transforms import (
    generate_detection_train_transform,
    generate_detection_val_transform,
)
from torch.utils.tensorboard import SummaryWriter
from visualize_image import visualize_one_xy_slice_in_3d_image
from warmup_scheduler import GradualWarmupScheduler

import monai
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.data import DataLoader, Dataset, box_utils, load_decathlon_datalist
from monai.data.utils import no_collation
from monai.networks.nets import resnet
from monai.transforms import ScaleIntensityRanged
from monai.utils import set_determinism


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="whether to print verbose detail during training, recommand True when you are not sure about hyper-parameters",
    )

    args = parser.parse_args()

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(args.returned_layers) + 1)],
        base_anchor_shapes=args.base_anchor_shapes,
    )

    conv1_t_size = [max(7, 2 * s + 1) for s in args.conv1_t_stride]
    backbone = resnet.ResNet(
        block=resnet.ResNetBottleneck,
        layers=[3, 4, 6, 3],
        block_inplanes=resnet.get_inplanes(),
        n_input_channels=args.n_input_channels,
        conv1_t_stride=args.conv1_t_stride,
        conv1_t_size=conv1_t_size,
    )
    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=args.spatial_dims,
        pretrained_backbone=False,
        trainable_backbone_layers=None,
        returned_layers=args.returned_layers,
    )
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    size_divisible = [s * 2 * 2 ** max(args.returned_layers) for s in feature_extractor.body.conv1.stride]
    net = torch.jit.script(
        RetinaNet(
            spatial_dims=args.spatial_dims,
            num_classes=len(args.fg_labels),
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=size_divisible,
        )
    )

    if device.type == 'cuda':
        DataParallel(net, device_ids=os.environ['CUDA_VISIBLE_DEVICES'])


    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.model_path)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])

    print(start_epoch)

if __name__ == '__main__':
    main()