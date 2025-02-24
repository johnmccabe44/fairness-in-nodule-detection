# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
from pathlib import Path
import sys

import monai
import numpy as np
import torch
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated, AffineBoxToWorldCoordinated, BoxToMaskd,
    ClipBoxToImaged, ConvertBoxModed, ConvertBoxToStandardModed, MaskToBoxd,
    RandCropBoxByPosNegLabeld, RandFlipBoxd, RandRotateBox90d, RandZoomBoxd,
    StandardizeEmptyBoxd)
from monai.data import DataLoader, Dataset, load_decathlon_datalist
from monai.data.utils import no_collation
from monai.transforms import (Compose, DeleteItemsd, EnsureChannelFirstd,
                              EnsureTyped, LoadImaged, MapTransform,
                              Orientationd, RandAdjustContrastd,
                              RandGaussianNoised, RandGaussianSmoothd,
                              RandRotated, RandScaleIntensityd,
                              RandShiftIntensityd, SaveImaged,
                              ScaleIntensityRanged, Transform)


class MultiBoxToMaskd(MapTransform):
    """
    Generate a mask from multiple boxes, where each box's value in the mask corresponds to its label.
    """
    def __init__(self, box_keys, label_keys, box_mask_keys, box_ref_image_keys):
        super().__init__(keys=[box_keys, label_keys, box_ref_image_keys])
        self.box_keys = box_keys
        self.label_keys = label_keys
        self.box_mask_keys = box_mask_keys
        self.box_ref_image_keys = box_ref_image_keys

    def __call__(self, data):
        d = dict(data)
        boxes = d[self.box_keys]
        labels = d[self.label_keys]
        ref_image = d[self.box_ref_image_keys]

        # Initialize the mask with the same spatial size as the reference image
        spatial_size = ref_image.shape[1:]  # Exclude channel dimension
        mask = np.zeros(spatial_size, dtype=np.int64)

        # Iterate through all boxes and labels
        for i, (box, label) in enumerate(zip(boxes, labels)):
            # Create a temporary binary mask for the current box
            box = np.round(box).astype(int)  # Ensure integer values
            x_min, y_min, z_min, x_max, y_max, z_max = box
            mask[x_min:x_max, y_min:y_max, z_min:z_max] = label


        # Convert mask to MetaTensor and attach the same meta as the reference image
        mask = np.expand_dims(mask, axis=0)
        mask = monai.data.MetaTensor(mask, meta=d[f"{self.box_ref_image_keys}"].meta)
        d[self.box_mask_keys] = mask

        return d



def main():
    parser = argparse.ArgumentParser(description="LUNA16 Detection Image Resampling")

    parser.add_argument(
        "-d",
        "--data-base-dir",
        default="/home/jmccabe/Projects/SOTAEvaluationNoduleDetection/cache/sota/lsut/masks",
        help="directory of the data",
    )

    parser.add_argument(
        "-o",
        "--orig-data-base-dir",
        default="/home/jmccabe/Projects/SOTAEvaluationNoduleDetection/cache/sota/lsut/detection",
        help="directory of the original data",
    )

    parser.add_argument(
        "-l",
        "--data-list-file-path",
        default="/home/jmccabe/Projects/SOTAEvaluationNoduleDetection/models/detection/datasplits/lsut/box_mask_temp.json",
        help="data list json file",
    )

    args = parser.parse_args()

    # 1. define transform
    # resample images to args.spacing defined in args.config_file.
    process_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image", "box"], dtype=torch.float32),
            EnsureTyped(keys=["image"], dtype=torch.long),
            StandardizeEmptyBoxd(box_keys=["box"], box_ref_image_keys="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024,
                a_max=300.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image"], dtype=torch.float16),
            EnsureTyped(keys=["label"], dtype=torch.long),
            ConvertBoxToStandardModed(box_keys=["box"], mode="cccwhd"),
            AffineBoxToImageCoordinated(
                box_keys=["box"],
                box_ref_image_keys="image",
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=True,
            ),
            MultiBoxToMaskd(
                box_keys="box",
                label_keys="label",
                box_mask_keys="box_mask",
                box_ref_image_keys="image"
            ),
            EnsureTyped(keys=["box_mask"], dtype=torch.float16),
        ]
    )
    # saved images to Nifti
    post_transforms = Compose(
        [
            SaveImaged(
                keys="box_mask",
                output_dir=args.data_base_dir,
                output_postfix="box_mask",
                resample=False,
            ),
        ]
    )

    # 2. prepare data
    for data_list_key in ["training", "validation", "test"]:
        # create a data loader
        process_data = load_decathlon_datalist(
            args.data_list_file_path,
            is_segmentation=True,
            data_list_key=data_list_key,
            base_dir=args.orig_data_base_dir,
        )
        process_ds = Dataset(
            data=process_data,
            transform=process_transforms,
        )
        process_loader = DataLoader(
            process_ds,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            collate_fn=no_collation,
        )

        print("-" * 10)
        for batch_data in process_loader:
            for batch_data_i in batch_data:

                for box in batch_data_i["box"]:
                    print(box)

                batch_data_i = post_transforms(batch_data_i)



if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
