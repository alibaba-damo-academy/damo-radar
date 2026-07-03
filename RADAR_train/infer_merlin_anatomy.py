import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
from monai import transforms
from monai.data.utils import dense_patch_slices
from typing import Any, Callable, List, Sequence, Tuple, Union
import datetime

from lavis.common.config import Config
from lavis.common.registry import registry
from lavis.common.dist_utils import get_rank, init_distributed_mode
import SimpleITK as sitk
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass



@dataclass
class DiseasePrompts:
    disease_name: str # name of disease
    region: str # region of the body where the disease is located
    positive_prompts: List[str] # list of regex expressions to extract positive cases
    negative_prompts: List[str] # list of regex expressions to extract negative cases

disease_prompts = {}

prompts = DiseasePrompts(
    disease_name="submucosal_edema",
    region="large bowel",
    positive_prompts=[
                      "mild diffuse submucosal edema",
                      "mild diffuse submucoasal edema",
                      "scattered areas of submucosal edema",
                      "mild submucosal edema",
                      "with submucosal edema",
                      "demonstrates submucosal edema",
                      "marked submucosal edema",
                      "there is submucosal edema",
                      "diffuse submucosal edema",
                    ],
    negative_prompts=[
                      "bowel : normal",
                      "no submucosal edema",
                    ],
)
disease_prompts["submucosal_edema"] = prompts

prompts = DiseasePrompts(
    disease_name="renal_hypodensities",
    region="kidney",
    positive_prompts=[
                      "kidneys and ureters : subcentimeter hypodensities",
                      "kidneys and ureters : 2 subcentimeter hypodensities",
                      "kidneys and ureters : two hypodensities",
                      "kidneys and ureters : small hypodensities",
                      "kidneys and ureters : hypodensities",
                      "bilateral renal hypodensities",
                      "subcentimeter renal hypodensities",
                      "multiple renal hypodensities",
                      "right renal hypodensities",
                      "left renal hypodensities",
                    ],
    negative_prompts=[
                      " "
                      "kidneys and ureters : normal",
                    ],
)
disease_prompts["renal_hypodensities"] = prompts

prompts = DiseasePrompts(
    disease_name="aortic_valve_calcification",
    region="heart",
    positive_prompts=[
                      "aortic valvular calcification",
                      "coronary artery and aortic valvular calcifications",
                      "aortic valve calcification",
                    ],
    negative_prompts=["vasculature : normal",
                    ],
)
disease_prompts["aortic_valve_calcification"] = prompts


prompts = DiseasePrompts(
    disease_name="pancreatic atrophy",
    region="pancreas",
    positive_prompts=[
                      "parenchymal atrophy",
                      "renal atrophy",
                      "pancreatic atrophy",
                      "atrophy of the pancreas",
                      "pancreas : severe atrophy",
                      "pancreas : diffuse atrophy",
                      "pancreas : fatty atrophy",
                      "pancreas : mild fatty atrophy",
                      "pancreas : diffuse fatty atrophy",
                    ],
    negative_prompts=["pancreas : normal",
                    ],
)
disease_prompts["pancreatic_atrophy"] = prompts

prompts = DiseasePrompts(
    disease_name="renal_cyst",
    region="kidney",
    positive_prompts=[
                      "renal cyst",
                      "bilateral renal cysts",
                      "simple renal cyst",
                      "multiple renal cysts",
                      "left renal cyst",
                      "right renal cyst",
                      "represent renal cyst",
                      "representing renal cyst",
                      "reflect renal cyst",
                    ],
    negative_prompts=["kidneys : normal",
                      "kidneys and ureters : normal",
                    ],
)
disease_prompts["renal_cyst"] = prompts

# prompts = DiseasePrompts(
#     disease_name="surgically_absent_gallbladder",
#     region="gallbladder",
#     positive_prompts=[
#                       "gallbladder : surgically absent",
#                     ],
#     negative_prompts=["gallbladder : normal",
#                     ],
# )
# disease_prompts["surgically_absent_gallbladder"] = prompts

prompts = DiseasePrompts(
    disease_name="atelectasis",
    region="lung",
    positive_prompts=[
                      "lower lobe atelectasis",
                      "bibasilar atelectasis",
                      "basilar passive atelectasis",
                      "mild bibasilar dependent atelectasis",
                      "mild dependent atelectasis",
                      "compatible with atelectasis",
                      "consistent with atelectasis",
                    ],
    negative_prompts=["lower thorax : normal .",
                    ],
)
disease_prompts["atelectasis"] = prompts

prompts = DiseasePrompts(
    disease_name="aortic aneurysm",
    region="aorta",
    positive_prompts=[
                      "infrarenal abdominal aortic aneurysm",
                      "the abdominal aortic aneurysm",
                      "abdominal aortic aneurysm , measuring",
                      "abdominal aortic aneurysm with",
                      "aortic aneurysm measures",
                      "aortic aneurysm measuring",
                      "aortic aneurysm with",
                    ],
    negative_prompts=["no aortic aneurysm",
                      "no abdominal aortic aneurysm",
                      "without abdominal aortic aneurysm",
                    ],
)
disease_prompts["aortic_aneurysm"] = prompts

prompts = DiseasePrompts(
    disease_name="hiatal_hernia",
    region="esophagus",
    positive_prompts=["small hiatal hernia",
                      "moderate sized hiatal hernia",
                      "moderate hiatal hernia",
                      "large hiatal hernia",
                      "small hiatus hernia",
                      ],
    negative_prompts=["no hernia",
                      "abdominal wall : normal .",
                    #   "gastrointestinal tract : normal ."
                    ],
)
disease_prompts["hiatal_hernia"] = prompts

prompts = DiseasePrompts(
    disease_name="biliary ductal dilation",
    region="liver",
    positive_prompts=[
                      "moderate biliary ductal dilation",
                      "mild biliary ductal dilation",
                      "severe biliary ductal dilation",
                      "mild intrahepatic biliary ductal dilation",
                      "severe intrahepatic biliary ductal dilation",
                      "moderate intrahepatic biliary ductal dilation",
                      "mild extrahepatic biliary ductal dilation",
                      "severe extrahepatic biliary ductal dilation",
                      "moderate extrahepatic biliary ductal dilation",
                    ],
    negative_prompts=["no biliary ductal dilation",
                      "no intrahepatic biliary ductal dilation",
                      "no extrahepatic biliary ductal dilation",
                      "no intra - or extrahepatic biliary ductal dilation",
                      "no intrahepatic or extrahepatic biliary ductal dilation"
                    ],
)
disease_prompts["biliary_ductal_dilation"] = prompts

prompts = DiseasePrompts(
    disease_name="cardiomegaly",
    region="heart",
    positive_prompts=["severe cardiomegaly",
                      "marked cardiomegaly",
                      "moderate cardiomegaly",
                      "mild cardiomegaly",
                      ". cardiomegaly",
                      "the heart is enlarged",
                    ],
    negative_prompts=["no cardiomegaly",
                      "the heart is normal in size",
                    ],
)
disease_prompts["cardiomegaly"] = prompts

prompts = DiseasePrompts(
    disease_name="splenomegaly",
    region="spleen",
    positive_prompts=[
                      "severe splenomegaly",
                      "mild splenomegaly",
                      "marked splenomegaly",
                      "spleen : splenomegaly",
                      "ongoing splenomegaly",
                    ],
    negative_prompts=["no splenomegaly",
                      "spleen : normal .",
                      "negative for splenomegaly",
                    ],
)
disease_prompts["splenomegaly"] = prompts

prompts = DiseasePrompts(
    disease_name="hepatomegaly",
    region="liver",
    positive_prompts=[
                      "massive hepatomegaly",
                      "marked hepatomegaly",
                      "stable hepatomegaly",
                      "mild hepatomegaly",
                      "liver and biliary tree : hepatomegaly",
                    ],
    negative_prompts=["liver and biliary tree : normal .",
                      "no hepatomegaly",
                    ],
)
disease_prompts["hepatomegaly"] = prompts


prompts = DiseasePrompts(
    disease_name="atherosclerosis",
    region="aorta",
    positive_prompts=[
        "vasculature : atherosclerosis",
        "mild atherosclerosis",
        "moderate atherosclerosis",
        "severe atherosclerosis",
        "focal atherosclerosis",
        "marked atherosclerosis",
        "calcific atherosclerosis",
        # "atherosclerosis is present",
    ],
    negative_prompts=[
        "no evidence of atherosclerosis",
        "no significant atherosclerosis",
    ],
)
disease_prompts["atherosclerosis"] = prompts

pleural_effusion_prompts = DiseasePrompts(
    disease_name="pleural effusion",
    region="lung",
    positive_prompts=["left pleural effusion", "right pleural effusion",
                      "bilateral pleural effusion",
                      "moderate pleural effusion",
                      "small pleural effusion",
                      ],
    negative_prompts=["no pleural effusion",
                      "without pleural effusion",
                      "no evidence of pleural effusion",
                      "no left pleural effusion",
                      "no right pleural effusion",
                      "no consolidation or pleural effusion",
                      "no pericardial or pleural effusion",
                      ]
)
disease_prompts["pleural_effusion"] = pleural_effusion_prompts

hepatic_steatosis_prompts = DiseasePrompts(
    disease_name="hepatic steatosis",
    region="liver",
    positive_prompts=["mild hepatic steatosis", "severe hepatic steatosis",
                      "moderate hepatic steatosis",
                      "diffuse hepatic steatosis",
                      "with hepatic steatosis",
                      "liver and biliary tree : hepatic steatosis",
                      "hepatic steatosis is noted",
                      ],
    negative_prompts=["no hepatic steatosis",
                      "without hepatic steatosis",
                      "no evidence of hepatic steatosis",
                     ],
)
disease_prompts["hepatic_steatosis"] = hepatic_steatosis_prompts

appendicitis_prompts = DiseasePrompts(
    disease_name="appendicitis",
    region="large bowel",
    positive_prompts=["consistent with acute appendicitis",
                      "consistent with appendicitis",
                      "compatible with appendicitis",
                      "compatible with acute uncomplicated appendicitis",
                      "compatible with uncomplicated appendicitis",
                      "compatible with acute appendicitis",
                      "represents early acute appendicitis",
                      "acute uncomplicated appendicitis",
                      "abdominal suggest acute appendicitis",
                      "concerning for appendicitis",
                      "perforated appendicitis",
                      #"suggest acute appendicitis", # to include these, need to avoid e.g. "there are no signs" earlier in the sentence
                      #"suggest appendicitis",
                      ],
    negative_prompts=["no evidence of acute appendicitis",
                      "no evidence of appendicitis",
                      "no appendicitis",
                      "no acute appendicitis",
                      "negative for appendicitis",
                      "without evidence of appendicitis",
                      "no sign of acute appendicitis",
                      "no secondary signs of acute appendicitis",
                      "no secondary signs of appendicitis",
                      ],
)
disease_prompts["appendicitis"] = appendicitis_prompts

gallstones_prompts = DiseasePrompts(
    disease_name="gallstones",
    region="gallbladder",
    positive_prompts=["gallstones present", "gallstones without ct findings of cholecystitis",
                      "gallstones are seen",
                      "gallstones are present",
                      "gallstones demonstrated",
                      "gallstones evident",
                      "large gallstones",
                      "small gallstones",
                      "multiple gallstones",
                      "multiple hyperdense gallstones",
                    "multiple calcified gallstones",
                    "multiple radiopaque gallstones",
                    "multiple layering gallstones",
                    "gallstones are again noted",
                    "gallstones are noted",
                      ],
    negative_prompts=["no gallstones", "no radiopaque gallstones", "no gallstones identified",
                      "without evidence of radiopaque gallstones",
                      "no evidence of radiopaque gallstones",
                      "without ct evidence of gallstones",
                      "no ct evidence of gallstones",
                      "no radiodense gallstones",
                      "without gallstones",
                    "without radiopaque gallstones",
                    "no calcified gallstones",
                    "no radiolucent gallstones",
                    "no sludge or gallstones",
                      ]
)
disease_prompts["gallstones"] = gallstones_prompts

hydronephrosis_prompts = DiseasePrompts(
    disease_name="hydronephrosis",
    region="kidney",
    positive_prompts=[
        "sided hydronephrosis",
        "bilateral hydronephrosis",
        "increase in mild hydronephrosis",
        "development of mild hydronephrosis",
        "is mild left hydronephrosis",
        "is mild right hydronephrosis",
        "is moderate left hydronephrosis",
        "is moderate right hydronephrosis",
        "moderate degree of hydronephrosis",
        ": left hydronephrosis",
        ": mild hydronephrosis",
        ": mild left hydronephrosis",
        ": mild right hydronephrosis",
        "there is hydronephrosis",
        "persistent moderate right hydronephrosis",
    ],
    negative_prompts=[
        "no hydronephrosis",
        "no focal hydronephrosis",
        "no renal hydronephrosis",
        "no evidence of hydronephrosis",
        "without hydronephrosis",
        "or hydronephrosis",
        "hydronephrosis or",
        "no evidence of hydronephrosis",
    ],
)
disease_prompts["hydronephrosis"] = hydronephrosis_prompts


bowel_obstruction_prompts = DiseasePrompts(
    disease_name="bowel obstruction",
    region="small bowel",
    positive_prompts=[
        "partial small bowel obstruction",
        "compatible with small bowel obstruction",
        "concerning for a small bowel obstruction",
        "consistent with small bowel obstruction",
        "mechanical small bowel obstruction",
        "high grade distal small bowel obstruction",
    ],
    negative_prompts=[
        "no bowel obstruction",
        "no small bowel obstruction",
        "no critical bowel obstruction",
        "no small or large bowel obstruction",
        "no evidence of bowel obstruction",
        "no evidence for bowel obstruction",
        "no evidence of small bowel obstruction",
        "no associated bowel obstruction",
        "negative for bowel obstruction",
        "no ct evidence of bowel obstruction",
        "no findings of bowel obstruction",
    ],
)
disease_prompts["bowel_obstruction"] = bowel_obstruction_prompts

fracture_prompts = DiseasePrompts(
    disease_name="fracture",
    region="lumbar vertebrae",
    positive_prompts=[
        "compression fracture",
        "fracture identified",
        "fractures identified",
        "rib fracture",
        "sacral fracture",
        "femoral fracture",
        "right iliac wing fracture",
        "musculoskeletal : nondisplaced fracture",
        "musculoskeletal : fracture",
    ],
    negative_prompts=[
        "no fracture",
        "no displaced fracture",
        "no acute fracture",
        "no evidence of fracture",
        "without evidence of fracture",
    ],
)
disease_prompts["fracture"] = fracture_prompts




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

def collate_fn(batch):
    return batch[0]

@torch.no_grad()
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list

def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
        """
        Compute scan interval according to the image size, roi size and overlap.
        Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
        use 1 instead to make sure sliding window works.

        """
        if len(image_size) != num_spatial_dims:
            raise ValueError("image coord different from spatial dims.")
        if len(roi_size) != num_spatial_dims:
            raise ValueError("roi coord different from spatial dims.")

        scan_interval = []
        for i in range(num_spatial_dims):
            if roi_size[i] == image_size[i]:
                scan_interval.append(int(roi_size[i]))
            else:
                interval = int(roi_size[i] * (1 - overlap))
                scan_interval.append(interval if interval > 0 else 1)
        return tuple(scan_interval)

def center_crop(image, mask, crop_size):
    x_min, y_min, z_min, x_max, y_max, z_max = masks_to_boxes_3d(mask)[0].long()
    
    crop_d, crop_h, crop_w = max(crop_size[0], z_max - z_min), max(crop_size[1], y_max - y_min), max(crop_size[2], x_max - x_min)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    cz = (z_min + z_max) // 2
    
    d, h, w = image.shape[-3:]

    x_start = max(0, cx - crop_w // 2)
    x_end = min(w, x_start + crop_w)
    if x_end - x_start < crop_w:
        x_start = max(0, x_end - crop_w)
    
    y_start = max(0, cy - crop_h // 2)
    y_end = min(h, y_start + crop_h)
    if y_end - y_start < crop_h:
        y_start = max(0, y_end - crop_h)
    
    z_start = max(0, cz - crop_d // 2)
    z_end = min(d, z_start + crop_d)
    if z_end - z_start < crop_d:
        z_start = max(0, z_end - crop_d)
    
    return image[..., z_start:z_end, y_start:y_end, x_start:x_end], mask[..., z_start:z_end, y_start:y_end, x_start:x_end]

class DataFolder(Dataset):
    def __init__(self):
        super().__init__()
        
        img_dir = '/mnt/nas/xiuxia/data/VLP/Merlin/download/merlinabdominalctdataset/merlin_data'
        if not os.path.exists(img_dir):
            assert False, 'Please modify the img_dir to your own path.'
        
        merlin_info = json.load(open('../merlin_report.json'))
        patient_list = []
        for id,minfo in merlin_info.items():
            if minfo['split'] == 'test':
                patient_list.append(id+'.nii.gz')

        self.img_paths = [
            os.path.join(img_dir, p)
            for p in patient_list
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
            transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
        ])
        
        self.pad_func = transforms.SpatialPadd(
            keys=["image"], 
            spatial_size=(96, 256, 384), 
            mode='constant',
            constant_values=0,
            method="end"
        )

        self.test_items = []
        self.test_organs = ['aorta', 'small bowel', 'large bowel', 'pancreas', 'liver', 'lumbar vertebrae', 'rib', 'thoracic vertebrae', 'heart', 'spleen', 'lung', 'esophagus', 'gallbladder', 'kidney']

        if dist.is_initialized():
            self.img_paths = self.img_paths[dist.get_rank()::dist.get_world_size()]
        
    @staticmethod
    def refine_prompt(prompt):
        prompt = re.sub(
            r"\s{2,}",
            " ",
            prompt
        )
        prompt = prompt.lower()
        prompt = prompt.rstrip("\n")
        prompt = prompt.strip(" ")
        if prompt[-1] != '.':
            prompt += '.'
        
        return prompt

    @staticmethod
    def get_spacing_letter(image_path):
        import re
        img_name = image_path.split('/')[-1]
        
        match = re.search(r"_z(\d+\.\d+)", img_name)
        z_spacing = float(match.group(1))
        
        pattern = r"_([AVDR]|NC)_"
        match = re.search(pattern, img_name)
        letter = match.group(1)

        return z_spacing, letter

    @staticmethod
    def get_patient_id(image_path):
        img_name = image_path.split('/')[-1]
        patient_id = img_name.split('_')[0]
        return patient_id
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # load image
        image_path = self.img_paths[index]
        data = {"image": image_path}
        try:
            res = transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True)(data)
        except:
            print(image_path) 
        image = res["image"]
        
        # reszie
        affine = res["image_meta_dict"]["affine"]
        spacing = (abs(affine[0, 0].item()), abs(affine[1, 1].item()), abs(affine[2, 2].item()))
        _, h, w, d = image.shape

        ref_spacing = (1.0, 1.0, 5.0)
        scale = [spacing[i] / ref_spacing[i] for i in range(3)]
        target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]

        trans = transforms.Compose(
            [
                transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear"),
                transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
            ]
        )
        resized_data = trans(res)

        # clip, normalize to [0, 1], crop non-zero region, pad to [96, 256, 384]
        # Our default setting is [-300, 400], following merlin's setting, we used [-1000, 1000]
        img_resized = resized_data["image"]
        image = img_resized
        image[image>1000] = 1000
        image[image<-1000] = -1000
        image = (image - image.min()) / (image.max()- image.min())
        img = image
        
        # crop non-zero region in image
        roi_coords = np.nonzero(img[0])
        min_dhw = torch.from_numpy(np.min(roi_coords, axis=1))
        max_dhw = torch.from_numpy(np.max(roi_coords, axis=1))

        extend_d = 5
        extend_hw = 20

        min_dhw = torch.max(
            min_dhw - torch.tensor([extend_d, extend_hw, extend_hw]),
            torch.tensor([0, 0, 0]),
        )
        max_dhw = torch.min(
            max_dhw + torch.tensor([extend_d, extend_hw, extend_hw]),
            torch.tensor([img.shape[1], img.shape[2], img.shape[3]]),
        )

        # pad data to [96, 256, 384] if smaller
        data["image"] = img[
            :, min_dhw[0] : max_dhw[0], min_dhw[1] : max_dhw[1], min_dhw[2] : max_dhw[2]
        ]
        
        data_pad = self.pad_func(data)
        data = data_pad
        
        file_name = image_path.split('/')[-1]
        patient_id = file_name.split('_')[0]
        test_organ_names = self.test_organs

        meta_info = {
            'file_name': file_name,
            'img_path': image_path,
            'patient_id': patient_id,
            'test_organ_names': test_organ_names,
            'letter': 'None'
        }
        return data['image'].as_tensor(), self.test_items, meta_info

def parse_args():
    parser = argparse.ArgumentParser(description="infer")
    parser.add_argument("--cfg-path", required=False, default='radar_config.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    return args

@torch.inference_mode()
def evaluate():
    args = parse_args()
    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)

    datafolder = DataFolder()
    dataloader = DataLoader(
        datafolder,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        drop_last=False,
        collate_fn=collate_fn
    )

    pad_func = transforms.DivisiblePadd(
                    keys=["image", "label"], 
                    k=32,
                    mode='constant', 
                    constant_values=0,
                    method="end"
            )
    
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)
    
    Cur_Time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if dist.is_initialized():
        print(f'Cur_Time: {Cur_Time}, Rank: {dist.get_rank()}')
    else:
        print(f'Cur_Time: {Cur_Time}')
    
    ckp_dir = './'
    ckpt_path = os.path.join(ckp_dir, f'checkpoint_radar_plus.pth')
    print('--> ckpt_path: ', ckpt_path)
    ckpt = torch.load(
        ckpt_path, map_location='cpu'
    )
    Cur_Time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if dist.is_initialized():
        print(f'Cur_Time: {Cur_Time}, Rank: {dist.get_rank()}, CPU load checkpoint done.')
    else:
        print(f'Cur_Time: {Cur_Time}, CPU load checkpoint done.')

    msg = model.load_state_dict(ckpt['model'], strict=False)
    print('--> missing_keys: ', [p for p in msg.missing_keys if 'text_encoder_m' not in p and 'text_proj_m' not in p])
    Cur_Time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if dist.is_initialized():
        print(f'Cur_Time: {Cur_Time}, Rank: {dist.get_rank()}, Model load checkpoint done.')
    else:
        print(f'Cur_Time: {Cur_Time}, Model load checkpoint done.')

    rank = get_rank()
    torch.cuda.set_device(rank)

    model.eval()
    model.cuda()

    sw_batch_size = 4
    overlap = 0.25
    roi_size = (96, 256, 384)

    miss_num = 0
    results = []
    organ_status = {}

    # --> get text feat
    num_positive_prompts = {}
    disease_prompts_new = {}
    for disease in disease_prompts:
        # add 'normal' as negative prompts
        region = disease_prompts[disease].region
        disease_new = disease.replace('_', '-')
        disease_prompts_new[(region, disease_new)] = {}
        disease_prompts_new[(region, disease_new)]['positive_prompts'] = [prompt.replace("\\", "") for prompt in disease_prompts[disease].positive_prompts]
        disease_prompts_new[(region, disease_new)]['negative_prompts'] = [prompt.replace("\\", "") for prompt in disease_prompts[disease].negative_prompts] + ['normal']
        num_positive_prompts[(region, disease_new)] = len(disease_prompts_new[(region, disease_new)]['positive_prompts'])
    model.num_positive_prompts = num_positive_prompts
    
    with torch.no_grad():
        text_feat_dict = {}
        for disease,prompts in disease_prompts_new.items():
            text = model.tokenizer(
                prompts['positive_prompts'] + prompts['negative_prompts'],
                padding="max_length",
                truncation=True,
                max_length=100,  # 100 is enough
                return_tensors="pt",
            ).to(model.device)

            text_output = model.text_encoder.forward_text(text)
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(model.text_proj(text_embeds[:, 0, :]), dim=-1)
            text_feat_dict[disease] = text_feat
            
    
    organ_feat_dict = {}
    save_path = os.path.join(ckp_dir, f'checkpoint_ZeroShotResult_anatomy.csv')
    
    for i, (image, test_items, meta_info) in enumerate(tqdm(dataloader, desc='Infer')):
        test_items = list(text_feat_dict.keys())
        skip_case = False
        for tmp_s in image.shape[1:]:
            if tmp_s > 1000:
                skip_case = True
                break
        if skip_case:
            continue
        
        fid = meta_info['file_name']
        organ_feat_dict[fid] = {}

        image = image[None].cuda()

        # We just test all intact organs from the mask predicted by model self.
        test_organs = meta_info['test_organ_names']

        image_size = list(image.shape[2:])
        num_spatial_dims = len(image.shape) - 2

        scan_interval = _get_scan_interval(
            image_size, roi_size, num_spatial_dims, overlap
        )
        slices = dense_patch_slices(image_size, roi_size, scan_interval)
        num_win = len(slices)
        organ_logits = dict(zip(test_items, [[] for _ in test_items]))

        # get full mask
        full_mask = torch.zeros((1, 37) + tuple(image_size)).cuda()
        count_map = torch.zeros_like(full_mask).cuda()

        for slice_g in range(0, num_win, sw_batch_size):
            slice_range = range(slice_g, min(slice_g + sw_batch_size, num_win))
            unravel_slice = [
                [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
                for idx in slice_range
            ]
            
            window_patches = torch.cat([image[win_slice] for win_slice in unravel_slice]).cuda()

            organ_logits, pred_window_seg_prob = model.forward_test_win(
                window_patches, 
                None,
                organ_logits,
                test_organs,
                text_feat_dict,
                organ_feat_dict[fid],
                None
            )

            # interpolate
            interpolated_seg_prob = F.interpolate(pred_window_seg_prob, size=window_patches.shape[2:], mode='trilinear')
            
            for ii, slice_idx in enumerate(slice_range):
                full_slice = unravel_slice[ii]
                full_mask[full_slice] += interpolated_seg_prob[ii]
                count_map[full_slice] += 1
        
        # Avoid division by zero by ensuring count_map is at least 1 everywhere
        count_map = torch.clamp(count_map, min=1)
        stitched_mask = full_mask / count_map  # argmax
        stitched_mask = stitched_mask.argmax(1).unsqueeze(0)

        # determin which organ still need to pred: compute intact oragn ids
        margin = 2
        boundaries = []
        squeeze_stitched_mask = stitched_mask.squeeze(0).squeeze(0)
        for d in range(squeeze_stitched_mask.dim()):
            start_slice = [slice(None)] * squeeze_stitched_mask.dim()
            end_slice = [slice(None)] * squeeze_stitched_mask.dim()
            
            start_slice[d] = slice(None, margin)
            end_slice[d] = slice(-margin, None)
            
            boundaries.append(squeeze_stitched_mask[tuple(start_slice)][squeeze_stitched_mask[tuple(start_slice)] > 0])
            boundaries.append(squeeze_stitched_mask[tuple(end_slice)][squeeze_stitched_mask[tuple(end_slice)] > 0])
        boundaries = torch.cat(boundaries)
        
        boundary_values = boundaries[boundaries > 0].flatten()
        boundary_organs = torch.unique(boundary_values)

        organ_ids, organ_counts = torch.unique(squeeze_stitched_mask, return_counts=True)
        organ_ids = organ_ids.long()
        organ_counts = organ_counts[organ_ids != 0]
        organ_ids = organ_ids[organ_ids != 0]

        # organs not touch boundary
        intact_organ_ids = [organ_id for organ_id, organ_count in zip(organ_ids, organ_counts) if organ_id not in boundary_organs]
        intact_organ_ids = torch.tensor(intact_organ_ids, device=squeeze_stitched_mask.device).long()
        intact_organ_ids = intact_organ_ids - 1

        for k, v in organ_logits.items():
            if not len(v):
                organ_name = k[0]
                organ_id = datafolder.organs.index(organ_name)

                window_patch, window_mask = center_crop(
                    image,
                    torch.eq(stitched_mask, organ_id + 1),
                    crop_size=roi_size
                )
                window_mask = window_mask.float()
                window_mask[window_mask == 1] = organ_id + 1

                pad_data = pad_func({'image': window_patch[0], 'label': window_mask[0]})
                window_patch, window_mask = pad_data['image'], pad_data['label']

                organ_logits, _ = model.forward_test_win(
                    window_patch[None], 
                    None,
                    organ_logits,
                    test_organs,
                    text_feat_dict,
                    organ_feat_dict[fid],
                    None,
                    skip_organ=organ_id
                )
        
        # Infer Entity: surgically_absent_gallbladder
        # gallbladder organ indice is 11
        if 11 not in torch.unique(stitched_mask):
            sag_prob = 0.
        else:
            sag_voxel_num  = (stitched_mask == 11).sum().item()
            sag_prob = sag_voxel_num
        
        res = [meta_info['file_name']] + [''] * len(test_items)
        organ_logits = {item: probs for item, probs in organ_logits.items() if len(probs) > 0}
        
        for item, probs in organ_logits.items():
            res[test_items.index(item) + 1] = np.concatenate(probs).mean(0)[1]  # get average of one organ in multi-widows
        res.append(sag_prob)  # add Entity: surgically_absent_gallbladder
        results.append(res)
    
    if dist.is_initialized():
        results = np.concatenate(all_gather(results), axis=0)
        organ_feat_dict = all_gather(organ_feat_dict)
    else:
        organ_feat_dict = [organ_feat_dict]
    
    if rank == 0:
        pd.DataFrame(
            results,
            columns=['file_name'] + [f'_'.join(k) for k in test_items]+['surgically_absent_gallbladder']
        ).to_csv(save_path, index=False, encoding='utf-8')
        
        print('Save result file successfully!')

if __name__ == '__main__':
    evaluate()
 
    if dist.is_initialized():  
        dist.destroy_process_group()
    
    torch.cuda.empty_cache()
