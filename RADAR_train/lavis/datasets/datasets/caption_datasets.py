"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re
import os
from collections import OrderedDict
from monai import transforms

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import numpy as np
from scipy import ndimage
import random
import torch
import torch.nn.functional as F
import h5py
import json
import SimpleITK as sitk

def masks_to_boxes_3d(masks):
    """Compute the bounding boxes around the provided 3D masks

    The masks should be in format [N, D, H, W] where N is the number of masks, (D, H, W) are the spatial dimensions.

    Returns a [N, 6] tensor, with the boxes in min_x, min_y, min_z, max_x, max_y, max_z format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device)

    d, h, w = masks.shape[-3:]

    z = torch.arange(0, d, dtype=torch.float, device=masks.device)
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)

    z, y, x = torch.meshgrid(z, y, x, indexing='ij')

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1).values
    x_min = x_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1).values
    y_min = y_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    z_mask = (masks * z.unsqueeze(0))
    z_max = z_mask.flatten(1).max(-1).values
    z_min = z_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )

class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.annotation = json.load(open('../ckpt/merlin_report_organ_report_v1.json'))
        self.organ_abnormal_info = json.load(open('../ckpt/merlin_report_organ_normal_v1.json'))
        
        vis_root = '../data/merlin_data_train_demo/resized_images'    # just resize spacing to [1, 1, 5]
        self.patient_paths = [
            os.path.join(vis_root, folder)
            for folder in os.listdir(vis_root)
        ]

        self.organs_cn = [
            '肾上腺', '主动脉', '竖脊肌', '脑', '锁骨', '大肠', '十二指肠', '食管', '面部', '股骨', 
            '胆囊', "臀肌", '心脏', '髋关节', '肱骨', '髂动脉', '髂静脉', '髂腰肌', '下腔静脉', '肾', 
            '肝', '肺', '胰腺', '门静脉', '肺动脉', '肋骨', '骶骨', '肩胛骨', '小肠', '脾', 
            '胃', '气管', '膀胱', '颈椎', '腰椎', '胸椎'
        ]
        
        organ_dict = {
            "肾上腺": "adrenal gland",
            "主动脉": "aorta",
            "竖脊肌": "erector spinae muscle",
            "脑": "brain",
            "锁骨": "clavicle",
            "大肠": "large bowel",
            "十二指肠": "duodenum",
            "食管": "esophagus",
            "面部": "face",
            "股骨": "femur",
            "胆囊": "gallbladder",
            "臀肌": "gluteus muscle",
            "心脏": "heart",
            "髋关节": "hip joint",
            "肱骨": "humerus",
            "髂动脉": "iliac artery",
            "髂静脉": "iliac vena",
            "髂腰肌": "iliopsoas muscle",
            "下腔静脉": "inferior vena cava",
            "肾": "kidney",
            "肝": "liver",
            "肺": "lung",
            "胰腺": "pancreas",
            "门静脉": "portal vein",
            "肺动脉": "pulmonary artery",
            "肋骨": "rib",
            "骶骨": "sacrum",
            "肩胛骨": "scapula",
            "小肠": "small bowel",
            "脾": "spleen",
            "胃": "stomach",
            "气管": "trachea",
            "膀胱": "bladder",
            "颈椎": "cervical vertebrae",
            "腰椎": "lumbar vertebrae",
            "胸椎": "thoracic vertebrae"
        }
        
        self.organs = [organ_dict[org] for org in self.organs_cn]
        

        self.loader = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),
            transforms.Transposed(keys=["image", "label"], indices=(0, 3, 2, 1)),
        ])
        self.pad_func = transforms.SpatialPadd(
            keys=["image", "label"],
            spatial_size=(96, 256, 384),
            mode='constant', 
            constant_values=0,
            method="end"
        )
        
        self.center_crop = transforms.CenterSpatialCropd(
            keys=["image", "label"], 
            roi_size=(96, 256, 384)
        )

        self.organ_ratios = {k: 1 for k in self.organs}

    def __getitem__(self, index):
        exit = False
        while not exit:
            try:
                patient_path = self.patient_paths[index]
                patient_id = patient_path.split('/')[-1].replace('.nii.gz', '')
                img_path = patient_path

                image_name = img_path.split('/')[-1].replace('.nii.gz', '')
                mask_path = img_path.replace('resized_images', 'resized_masks')

                data = self.loader({'image': img_path, 'label': mask_path})
                
                # --> center crop
                data = self.pad_func(data)
                data = self.center_crop(data)
                
                image = data['image']
                label = data['label']
                
                # --> online compute intact organ ids
                pul_seg = label[0]
                organ_ids = torch.unique(pul_seg)
                organ_ids = organ_ids.long()
                organ_ids = organ_ids[organ_ids != 0]
                boundaries = [
                    pul_seg[0], pul_seg[-1],
                    pul_seg[:, 0], pul_seg[:, -1],
                    pul_seg[:, :, 0], pul_seg[:, :, -1]
                ]
                non_zero_boundaries = [b[b != 0].flatten() for b in boundaries]
                boundary_values = torch.cat(non_zero_boundaries)
                boundary_organs = torch.unique(boundary_values)
                # 
                intact_organ_ids = [organ_id for organ_id in organ_ids if organ_id not in boundary_organs]
                intact_organ_ids = torch.tensor(intact_organ_ids).long()
                intact_organ_ids = intact_organ_ids - 1

                reported_organs_ids = [self.organs.index(organ) for organ in self.annotation[patient_id].copy().keys() if organ not in ['report', 'desc', 'conc']]
                reported_organs_ids = list(filter(lambda x: x + 1 in organ_ids, reported_organs_ids))
                reported_organs_ids = torch.tensor(reported_organs_ids).long()
                intact_organ_ids = torch.cat([intact_organ_ids, reported_organs_ids])
                intact_organ_ids = intact_organ_ids.unique()
                
                
                # --> online process image and mask
                # clip, normalize to [0, 1], crop non-zero region
                # Our default setting is [-300, 400], following merlin's setting, we used [-1000, 1000]
                image[image>1000] = 1000
                image[image<-1000] = -1000
                image = (image - image.min()) / (image.max()- image.min())
                
                data = {
                    'image': image,
                    'label': pul_seg.unsqueeze(0)
                }
                
                data = self.vis_processor(data)
                image = data['image'].as_tensor()
                pul_seg = data['label'][0].as_tensor()

                organ_captions = self.annotation[patient_id].copy()
                organ_captions = self.text_processor(organ_captions)

                pid_organ_abnormal_info = self.organ_abnormal_info[patient_id]
                organ_abnormal_flags = torch.zeros(len(self.organs), dtype=bool)
                for i, organ in enumerate(self.organs):
                    if organ in pid_organ_abnormal_info and pid_organ_abnormal_info[organ] == "abnormal":
                        organ_abnormal_flags[i] = True
                        
                    if organ not in organ_captions:
                        organ_captions[organ] = f'normal.'
                exit = True
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.patient_paths) - 1)
                continue
        
        return {
            "image": image,
            "seg": pul_seg,
            "text_input": organ_captions,
            "index": index,
            "letter": 'letter',
            "patient_id": patient_id,
            "organ_abnormal_flags": organ_abnormal_flags,
            "image_path": img_path,
        }

    @staticmethod
    def random_crop_with_bounding_box(image, pul_seg, roi_organ_id, crop_size):
        crop_z, crop_y, crop_x = crop_size

        mask = torch.eq(pul_seg, roi_organ_id + 1)

        if len(mask.size()) == 3:
            bounding_box = masks_to_boxes_3d(mask[None])[0]
        else:
            bounding_box = masks_to_boxes_3d(mask)[0]
        
        bb_x_min, bb_y_min, bb_z_min, bb_x_max, bb_y_max, bb_z_max = bounding_box.tolist()
        bb_width_x = bb_x_max - bb_x_min + 1
        bb_width_y = bb_y_max - bb_y_min + 1
        bb_width_z = bb_z_max - bb_z_min + 1

        stick_x = False
        if bb_width_x >= crop_x:
            bb_x_min = bb_x_min + random.randint(0, bb_width_x - crop_x)
            bb_x_max = bb_x_min + crop_x - 1
            bb_width_x = crop_x
            
            stick_x = True

        stick_y = False
        if bb_width_y >= crop_y:
            bb_y_min = bb_y_min + random.randint(0, bb_width_y - crop_y)
            bb_y_max = bb_y_min + crop_y - 1
            bb_width_y = crop_y
            
            stick_y = True

        stick_z = False
        if bb_width_z >= crop_z:
            bb_z_min = bb_z_min + random.randint(0, bb_width_z - crop_z)
            bb_z_max = bb_z_min + crop_z - 1
            bb_width_z = crop_z

            stick_z = True
        
        if not stick_x:
            crop_x_limit = np.arange(0, max(bb_x_min, 1)).astype(int)
            crop_x_limit = crop_x_limit[(crop_x_limit + crop_x > bb_x_max + 1) & (crop_x_limit + crop_x <= pul_seg.shape[2])]
            
            if len(crop_x_limit):
                start_x = np.random.choice(crop_x_limit)
                end_x = start_x + crop_x
            else:
                end_x = int(bb_x_max + 1)
                start_x = end_x - crop_x
        else:
            start_x = int(bb_x_min)
            end_x = int(bb_x_max + 1)
        
        if not stick_y:
            crop_y_limit = np.arange(0, max(bb_y_min, 1)).astype(int)
            crop_y_limit = crop_y_limit[(crop_y_limit + crop_y > bb_y_max + 1) & (crop_y_limit + crop_y <= pul_seg.shape[1])]

            if len(crop_y_limit):
                start_y = np.random.choice(crop_y_limit)
                end_y = start_y + crop_y
            else:
                end_y = int(bb_y_max + 1) 
                start_y = end_y - crop_y
        else:
            start_y = int(bb_y_min)
            end_y = int(bb_y_max + 1)
        
        if not stick_z:
            crop_z_limit = np.arange(0, max(bb_z_min, 1)).astype(int)
            crop_z_limit = crop_z_limit[(crop_z_limit + crop_z > bb_z_max + 1) & (crop_z_limit + crop_z <= pul_seg.shape[0])]
            
            if len(crop_z_limit):
                start_z = np.random.choice(crop_z_limit)
                end_z = start_z + crop_z
            else:
                end_z = int(bb_z_max + 1)
                start_z = end_z - crop_z
        else:
            start_z = int(bb_z_min)
            end_z = int(bb_z_max + 1)

        cropped_image = image[start_z:end_z, start_y:end_y, start_x:end_x]
        cropped_mask = pul_seg[start_z:end_z, start_y:end_y, start_x:end_x]

        return cropped_image, cropped_mask, stick_x or stick_y or stick_z

    @staticmethod
    def get_spacing_letter(image_path):
        img_name = image_path.split('/')[-1]

        match = re.search(r"_z(\d+\.\d+)", img_name)
        z_spacing = float(match.group(1))
        
        pattern = r"_([AVDR])_"
        match = re.search(pattern, img_name)
        letter = match.group(1)
        return z_spacing, letter

    @staticmethod
    def get_patient_id(image_path):
        img_name = image_path.split('/')[-1]
        
        if 'clear_normal' in img_name:
            patient_id = img_name.split('_')[0]
        elif 'normal_lung_eso_bre' in img_name:
            patient_id = img_name.split('.')[0]
        else:
            patient_id = '_'.join(str(img_name).split('_')[1:3])
        return patient_id
