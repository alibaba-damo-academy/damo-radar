import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
from monai import transforms
from monai.data.utils import dense_patch_slices
from typing import Any, Callable, List, Sequence, Tuple, Union
import datetime
import SimpleITK as sitk
import torch.nn.functional as F
from pathlib import Path
from copy import deepcopy
import torch.distributed as dist
from dynamic_network_architectures.med import XBertEncoder, XBertLMHeadDecoder
from dynamic_network_architectures.vision_branch import VisionBranch
from transformers import BertTokenizer


model_root = os.environ.get("MODEL_ROOT", "../ckpt")
configs_root = os.environ.get("CONFIGS_ROOT", "../ckpt")

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
    def __init__(self, img_dir):
        super().__init__()
        
        patient_list = os.listdir(img_dir)
        self.img_paths = [
            os.path.join(img_dir, p)
            for p in patient_list
        ]
        
        self.pad_func = transforms.SpatialPadd(
            keys=["image"], 
            spatial_size=(96, 256, 384), 
            mode='constant', 
            constant_values=0,
            method="end"
        )

        self.organs = [
            '肾上腺', '主动脉', '竖脊肌', '脑', '锁骨', '大肠', '十二指肠', '食管', '面部', '股骨', 
            '胆囊', "臀肌", '心脏', '髋关节', '肱骨', '髂动脉', '髂静脉', '髂腰肌', '下腔静脉', '肾', 
            '肝', '肺', '胰腺', '门静脉', '肺动脉', '肋骨', '骶骨', '肩胛骨', '小肠', '脾', 
            '胃', '气管', '膀胱', '颈椎', '腰椎', '胸椎'
        ]
        # organ_dict = {
        #     "肾上腺": "adrenal gland", "主动脉": "aorta", "竖脊肌": "erector spinae muscle", "脑": "brain", "锁骨": "clavicle", "大肠": "large bowel", "十二指肠": "duodenum", 
        #     "食管": "esophagus", "面部": "face", "股骨": "femur", "胆囊": "gallbladder", "臀肌": "gluteus muscle", "心脏": "heart", "髋关节": "hip joint", "肱骨": "humerus", 
        #     "髂动脉": "iliac artery", "髂静脉": "iliac vena", "髂腰肌": "iliopsoas muscle", "下腔静脉": "inferior vena cava", "肾": "kidney", "肝": "liver", "肺": "lung", 
        #     "胰腺": "pancreas", "门静脉": "portal vein", "肺动脉": "pulmonary artery", "肋骨": "rib", "骶骨": "sacrum", "肩胛骨": "scapula", "小肠": "small bowel", "脾": "spleen", 
        #     "胃": "stomach", "气管": "trachea", "膀胱": "bladder", "颈椎": "cervical vertebrae", "腰椎": "lumbar vertebrae", "胸椎": "thoracic vertebrae"
        # }
        
        self.test_items = ['主动脉_主动脉夹层', '主动脉_主动脉瘤', '主动脉_粥样硬化', '主动脉_钙化', '十二指肠_占位', '十二指肠_囊袋状突出影', '十二指肠_憩室', '十二指肠_梗阻', '十二指肠_溃疡', '大肠_克罗恩病', '大肠_大肠（壁）钙化', '大肠_急慢性（结）肠炎', '大肠_浆膜面毛糙', '大肠_溃疡性结肠炎', '大肠_直肠癌', '大肠_积液积气', '大肠_结肠癌', '大肠_肠壁毛糙', '大肠_肠壁水肿', '大肠_肠套叠', '大肠_肠憩室', '大肠_肠梗阻', '大肠_肠穿孔', '大肠_肠道扩张', '大肠_脂肪间隙模糊', '大肠_阑尾炎', '大肠_阑尾粪石', '小肠_克罗恩病', '小肠_套叠', '小肠_扭转', '小肠_梗阻', '小肠_淋巴瘤', '小肠_积气积液', '小肠_系膜指膜炎', '小肠_系膜淋巴结肿大', '小肠_肠壁增厚', '小肠_肠管扩张', '小肠_脂肪瘤', '小肠_间质瘤（胃肠间质瘤-gist）', '小肠_（急慢性）小肠炎', '心脏_心包积液', '心脏_心影（脏）增大', '肋骨_转移瘤（乳腺癌 骨转移）', '肋骨_骨折', '肋骨_骨质破坏', '肝_低密度影', '肝_格林森鞘积液', '肝_比例失调', '肝_波浪状改变', '肝_硬化', '肝_结节状强化', '肝_肝内胆管扩张', '肝_肝内胆管结石', '肝_肝内钙化灶', '肝_肝囊肿', '肝_肝细胞癌', '肝_肝胆管内高密度影', '肝_肝血管瘤', '肝_胆管癌', '肝_脂肪肝', '肝_脓肿', '肝_转移瘤', '肝_边缘不规则', '肺_斑片影', '肺_气胸', '肺_结节', '肺_肺占位', '肺_肺萎陷', '肺_胸腔积液', '肺_膨胀不全', '肺_转移瘤', '肺_钙化灶', '肺_高密度影', '肾_低密度影', '肾_囊肿', '肾_多囊肾', '肾_实质变薄', '肾_无强化囊性灶', '肾_肾动脉瘤', '肾_肾盂扩张', '肾_肾盂癌', '肾_肾盂积水', '肾_肾细胞癌（透明细胞癌）', '肾_肾萎缩', '肾_肾血管平滑肌脂肪瘤', '肾_肾（盂）结石', '肾_高密度影', '肾上腺_增生', '肾上腺_结节', '肾上腺_脂肪瘤', '肾上腺_腺瘤', '肾上腺_转移瘤', '肾上腺_钙化', '胃_壁水肿', '胃_扩张', '胃_胃底静脉曲张', '胃_胃溃疡', '胃_胃癌', '胃_间质瘤（gist）', '胆囊_结石', '胆囊_结节状致密影', '胆囊_胆囊增大', '胆囊_胆囊炎', '胆囊_胆囊癌', '胆囊_胆囊腺肌症', '胆囊_胆管壁增厚', '胆囊_胆管扩张', '胆囊_胆管炎', '胆囊_胆管癌', '胆囊_胆管积气', '胆囊_胆管结石', '胆囊_高密度影', '胆囊_黄色肉芽肿', '胰腺_低密度影', '胰腺_囊肿', '胰腺_围脂肪间隙模糊', '胰腺_肿瘤或胰腺癌', '胰腺_胰周假性囊肿', '胰腺_胰管扩张', '胰腺_胰管结石', '胰腺_胰腺炎', '胰腺_胰腺饱满', '胰腺_萎缩', '脾_低密度灶', '脾_副脾', '脾_囊肿', '脾_梗死', '脾_片状低密度区', '脾_脾大', '脾_脾脏淋巴瘤', '脾_钙化', '膀胱_憩室', '膀胱_结石', '膀胱_膀胱壁毛糙', '膀胱_膀胱炎', '膀胱_膀胱癌', '膀胱_软组织密度影', '门静脉_增宽', '门静脉_栓塞', '门静脉_高压', '食管_增粗迂曲血管影', '食管_管壁增厚', '食管_裂孔疝', '食管_静脉扩张迂曲', '食管_静脉曲张', '骶骨_骨炎']
        # english_mapping = {'主动脉_主动脉夹层': 'Aorta_Aortic dissection', '主动脉_主动脉瘤': 'Aorta_Aortic aneurysm', '主动脉_粥样硬化': 'Aorta_Atherosclerosis', '主动脉_钙化': 'Aorta_Calcification', '十二指肠_占位': 'Duodenum_Mass', '十二指肠_囊袋状突出影': 'Duodenum_Saccular outpouching', '十二指肠_憩室': 'Duodenum_Diverticulum', '十二指肠_梗阻': 'Duodenum_Obstruction', '十二指肠_溃疡': 'Duodenum_Ulcer', '大肠_克罗恩病': "Large bowel_Crohn's disease", '大肠_大肠（壁）钙化': 'Large bowel_Mural calcification', '大肠_急慢性（结）肠炎': 'Large bowel_Colitis', '大肠_浆膜面毛糙': 'Large bowel_Serosal surface irregularity', '大肠_溃疡性结肠炎': 'Large bowel_Ulcerative colitis', '大肠_直肠癌': 'Large bowel_Rectal cancer', '大肠_积液积气': 'Large bowel_Gas and fluid accumulation', '大肠_结肠癌': 'Large bowel_Colon cancer', '大肠_肠壁毛糙': 'Large bowel_Wall irregularity', '大肠_肠壁水肿': 'Large bowel_Wall edema', '大肠_肠套叠': 'Large bowel_Intussusception', '大肠_肠憩室': 'Large bowel_Diverticulum', '大肠_肠梗阻': 'Large bowel_Obstruction', '大肠_肠穿孔': 'Large bowel_Perforation', '大肠_肠道扩张': 'Large bowel_Dilatation', '大肠_脂肪间隙模糊': 'Large bowel_Blurring of fat planes', '大肠_阑尾炎': 'Large bowel_Appendicitis', '大肠_阑尾粪石': 'Large bowel_Appendicolith', '小肠_克罗恩病': "Small bowel_Crohn's disease", '小肠_套叠': 'Small bowel_Intussusception', '小肠_扭转': 'Small bowel_Volvulus', '小肠_梗阻': 'Small bowel_Obstruction', '小肠_淋巴瘤': 'Small bowel_Lymphoma', '小肠_积气积液': 'Small bowel_Gas and fluid accumulation', '小肠_系膜指膜炎': 'Small bowel_Mesenteric panniculitis', '小肠_系膜淋巴结肿大': 'Small bowel_Mesenteric lymphadenopathy', '小肠_肠壁增厚': 'Small bowel_Wall thickening', '小肠_肠管扩张': 'Small bowel_Dilatation', '小肠_脂肪瘤': 'Small bowel_Lipoma', '小肠_间质瘤（胃肠间质瘤-gist）': 'Small bowel_Gastrointestinal stromal tumor', '小肠_（急慢性）小肠炎': 'Small bowel_Enteritis', '心脏_心包积液': 'Heart_Pericardial effusion', '心脏_心影（脏）增大': 'Heart_Cardiomegaly', '肋骨_转移瘤（乳腺癌 骨转移）': 'Rib_Metastasis', '肋骨_骨折': 'Rib_Fracture', '肋骨_骨质破坏': 'Rib_Bone destruction', '肝_低密度影': 'Liver_Hypoattenuating lesion', '肝_格林森鞘积液': 'Liver_Periportal edema', '肝_比例失调': 'Liver_Lobar volume disproportion', '肝_波浪状改变': 'Liver_Undulating contour', '肝_硬化': 'Liver_Cirrhosis', '肝_结节状强化': 'Liver_Nodular enhancement', '肝_肝内胆管扩张': 'Liver_Intrahepatic bile duct dilatation', '肝_肝内胆管结石': 'Liver_Hepatolithiasis', '肝_肝内钙化灶': 'Liver_Intrahepatic calcification', '肝_肝囊肿': 'Liver_Cyst', '肝_肝细胞癌': 'Liver_Hepatocellular carcinoma', '肝_肝胆管内高密度影': 'Liver_Hyperattenuating lesion in intrahepatic bile ducts', '肝_肝血管瘤': 'Liver_Hemangioma', '肝_胆管癌': 'Liver_Intrahepatic cholangiocarcinoma', '肝_脂肪肝': 'Liver_Steatotic liver disease', '肝_脓肿': 'Liver_Abscess', '肝_转移瘤': 'Liver_Metastasis', '肝_边缘不规则': 'Liver_Irregular margin', '肺_斑片影': 'Lung_Patchy opacity', '肺_气胸': 'Lung_Pneumothorax', '肺_结节': 'Lung_Nodule', '肺_肺占位': 'Lung_Mass', '肺_肺萎陷': 'Lung_Pulmonary collapse', '肺_胸腔积液': 'Lung_Pleural effusion', '肺_膨胀不全': 'Lung_Atelectasis', '肺_转移瘤': 'Lung_Metastasis', '肺_钙化灶': 'Lung_Calcification', '肺_高密度影': 'Lung_Hyperattenuating opacity', '肾_低密度影': 'Kidney_Hypoattenuating lesion', '肾_囊肿': 'Kidney_Cyst', '肾_多囊肾': 'Kidney_Polycystic kidney disease', '肾_实质变薄': 'Kidney_Parenchymal thinning', '肾_无强化囊性灶': 'Kidney_Nonenhancing cystic lesion', '肾_肾动脉瘤': 'Kidney_Renal artery aneurysm', '肾_肾盂扩张': 'Kidney_Renal pelvic dilatation', '肾_肾盂癌': 'Kidney_Renal pelvic cancer', '肾_肾盂积水': 'Kidney_Hydronephrosis', '肾_肾细胞癌（透明细胞癌）': 'Kidney_Renal cell carcinoma', '肾_肾萎缩': 'Kidney_Atrophy', '肾_肾血管平滑肌脂肪瘤': 'Kidney_Angiomyolipoma', '肾_肾（盂）结石': 'Kidney_Nephrolithiasis', '肾_高密度影': 'Kidney_Hyperattenuating lesion', '肾上腺_增生': 'Adrenal gland_Hyperplasia', '肾上腺_结节': 'Adrenal gland_Nodule', '肾上腺_脂肪瘤': 'Adrenal gland_Lipoma', '肾上腺_腺瘤': 'Adrenal gland_Adenoma', '肾上腺_转移瘤': 'Adrenal gland_Metastasis', '肾上腺_钙化': 'Adrenal gland_Calcification', '胃_壁水肿': 'Stomach_Wall edema', '胃_扩张': 'Stomach_Dilatation', '胃_胃底静脉曲张': 'Stomach_Gastric fundal varices', '胃_胃溃疡': 'Stomach_Ulcer', '胃_胃癌': 'Stomach_Gastric cancer', '胃_间质瘤（gist）': 'Stomach_Gastrointestinal stromal tumor (GIST)', '胆囊_结石': 'Gallbladder_Cholecystolithiasis', '胆囊_结节状致密影': 'Gallbladder_Nodular stone-like hyperattenuating lesion', '胆囊_胆囊增大': 'Gallbladder_Distention', '胆囊_胆囊炎': 'Gallbladder_Cholecystitis', '胆囊_胆囊癌': 'Gallbladder_Gallbladder cancer', '胆囊_胆囊腺肌症': 'Gallbladder_Adenomyomatosis', '胆囊_胆管壁增厚': 'Gallbladder_Extrahepatic bile duct wall thickening', '胆囊_胆管扩张': 'Gallbladder_Extrahepatic bile duct dilatation', '胆囊_胆管炎': 'Gallbladder_Cholangitis', '胆囊_胆管癌': 'Gallbladder_Cholangiocarcinoma', '胆囊_胆管积气': 'Gallbladder_Pneumobilia', '胆囊_胆管结石': 'Gallbladder_Extrahepatic bile duct stone', '胆囊_高密度影': 'Gallbladder_Hyperattenuating lesion', '胆囊_黄色肉芽肿': 'Gallbladder_Xanthogranuloma', '胰腺_低密度影': 'Pancreas_Low-density lesion', '胰腺_囊肿': 'Pancreas_Cyst', '胰腺_围脂肪间隙模糊': 'Pancreas_Blurring of peripancreatic fat planes', '胰腺_肿瘤或胰腺癌': 'Pancreas_Pancreatic cancer', '胰腺_胰周假性囊肿': 'Pancreas_Peripancreatic pseudocyst', '胰腺_胰管扩张': 'Pancreas_Pancreatic duct dilatation', '胰腺_胰管结石': 'Pancreas_Pancreatic duct calculus', '胰腺_胰腺炎': 'Pancreas_Pancreatitis', '胰腺_胰腺饱满': 'Pancreas_Enlargement', '胰腺_萎缩': 'Pancreas_Atrophy', '脾_低密度灶': 'Spleen_Hypoattenuating lesion', '脾_副脾': 'Spleen_Accessory spleen', '脾_囊肿': 'Spleen_Cyst', '脾_梗死': 'Spleen_Infarction', '脾_片状低密度区': 'Spleen_Patchy hypoattenuating lesion', '脾_脾大': 'Spleen_Splenomegaly', '脾_脾脏淋巴瘤': 'Spleen_Lymphoma', '脾_钙化': 'Spleen_Calcification', '膀胱_憩室': 'Bladder_Diverticulum', '膀胱_结石': 'Bladder_Stone', '膀胱_膀胱壁毛糙': 'Bladder_Wall irregularity', '膀胱_膀胱炎': 'Bladder_Cystitis', '膀胱_膀胱癌': 'Bladder_Bladder cancer', '膀胱_软组织密度影': 'Bladder_Soft-tissue attenuation lesion', '门静脉_增宽': 'Portal vein_Dilatation', '门静脉_栓塞': 'Portal vein_Thrombosis', '门静脉_高压': 'Portal vein_Hypertension', '食管_增粗迂曲血管影': 'Esophagus_Dilated and tortuous tubular opacities', '食管_管壁增厚': 'Esophagus_Wall thickening', '食管_裂孔疝': 'Esophagus_Hiatal hernia', '食管_静脉扩张迂曲': 'Esophagus_Dilated and tortuous veins', '食管_静脉曲张': 'Esophagus_Varices', '骶骨_骨炎': 'Sacrum_Osteitis'}
        self.test_organs = list(set([item.split('_')[0] for item in self.test_items]))
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # load image
        image_path = self.img_paths[index]
        data = {"image": image_path}
        res = transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True)(data)
        image = res["image"]
        
        affine = res["image_meta_dict"]["affine"]
        spacing = (
            abs(affine[0, 0].item()),
            abs(affine[1, 1].item()),
            abs(affine[2, 2].item())
        )
        _, h, w, d = image.shape
        orig_shape_hwd = (h, w, d)

        ref_spacing = (1.0, 1.0, 5.0)
        scale = [spacing[i] / ref_spacing[i] for i in range(3)]
        target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]  # [H', W', D']

        trans = transforms.Compose(
            [
                transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear"),
                transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
            ]
        )
        resized_data = trans(res)

        img_resized = resized_data["image"]   # [C, D', W', H']
        image = img_resized
        image[image > 400] = 400
        image[image < -300] = -300
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        img = image

        # crop non-zero region in image
        roi_coords = np.nonzero(img[0].cpu().numpy())
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

        cropped_image = img[
            :,
            min_dhw[0]: max_dhw[0],
            min_dhw[1]: max_dhw[1],
            min_dhw[2]: max_dhw[2]
        ]
        crop_shape_dhw = tuple(cropped_image.shape[1:])

        # pad data to [96, 256, 384] if smaller
        data["image"] = cropped_image
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
            'letter': 'None',
        }
        return data['image'].as_tensor(), self.test_items, meta_info


class RADAR(nn.Module):
    def __init__(
        self,
        image_encoder,
        text_encoder,
        text_decoder=None,
        queue_size=1234,
        alpha=0.4,
        embed_dim=256,
        momentum=0.995,
        tie_enc_dec_weights=True,
        max_txt_len=175,
    ):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(configs_root, "bert-base-chinese"))

        text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        text_width = text_encoder.config.hidden_size
        vision_width = 256

        self.text_proj = nn.Linear(text_width, embed_dim)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.alpha = alpha
        self.max_txt_len = max_txt_len

        self.organs = [
            '肾上腺', '主动脉', '竖脊肌', '脑', '锁骨', '大肠', '十二指肠', '食管', '面部', '股骨', 
            '胆囊', "臀肌", '心脏', '髋关节', '肱骨', '髂动脉', '髂静脉', '髂腰肌', '下腔静脉', '肾', 
            '肝', '肺', '胰腺', '门静脉', '肺动脉', '肋骨', '骶骨', '肩胛骨', '小肠', '脾', 
            '胃', '气管', '膀胱', '颈椎', '腰椎', '胸椎'
        ]
        # organ_dict = {
        #     "肾上腺": "adrenal gland", "主动脉": "aorta", "竖脊肌": "erector spinae muscle", "脑": "brain", "锁骨": "clavicle", "大肠": "large bowel", "十二指肠": "duodenum", 
        #     "食管": "esophagus", "面部": "face", "股骨": "femur", "胆囊": "gallbladder", "臀肌": "gluteus muscle", "心脏": "heart", "髋关节": "hip joint", "肱骨": "humerus", 
        #     "髂动脉": "iliac artery", "髂静脉": "iliac vena", "髂腰肌": "iliopsoas muscle", "下腔静脉": "inferior vena cava", "肾": "kidney", "肝": "liver", "肺": "lung", 
        #     "胰腺": "pancreas", "门静脉": "portal vein", "肺动脉": "pulmonary artery", "肋骨": "rib", "骶骨": "sacrum", "肩胛骨": "scapula", "小肠": "small bowel", "脾": "spleen", 
        #     "胃": "stomach", "气管": "trachea", "膀胱": "bladder", "颈椎": "cervical vertebrae", "腰椎": "lumbar vertebrae", "胸椎": "thoracic vertebrae"
        # }
        
        self.attention = nn.MultiheadAttention(
            embed_dim=vision_width,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.vision_projs = nn.ModuleList([nn.Linear(vision_width, embed_dim) for _ in range(len(self.organs))])
        self.query_tokens = nn.Parameter(torch.zeros(len(self.organs), vision_width))

    @torch.inference_mode()
    def forward_test_win(
        self, 
        images, 
        masks, 
        organ_logits,
        test_organs,
        text_feat_dict,
        organ_feat_dict,
        whole_organ_sizes,
        skip_organ=None
    ):
        seg_probs, seg, image_embeds1, image_embeds2, image_embeds3, organ_token_flags1, organ_token_flags2, organ_token_flags3 = self.visual_encoder(images, None)

        margin = 2
        masks = seg
        
        for i, (embed1, embed2, embed3, mask) in enumerate(zip(image_embeds1, image_embeds2, image_embeds3, masks)):
            boundaries = []
            for d in range(mask.dim()):
                start_slice = [slice(None)] * mask.dim()
                end_slice = [slice(None)] * mask.dim()
                
                start_slice[d] = slice(None, margin)
                end_slice[d] = slice(-margin, None)
                
                boundaries.append(mask[tuple(start_slice)][mask[tuple(start_slice)] > 0])
                boundaries.append(mask[tuple(end_slice)][mask[tuple(end_slice)] > 0])
            boundaries = torch.cat(boundaries)
            
            boundary_values = boundaries[boundaries > 0].flatten()
            boundary_organs = torch.unique(boundary_values)

            if skip_organ is not None:
                boundary_organs = boundary_organs[boundary_organs != skip_organ + 1]
            
            organ_ids, organ_counts = torch.unique(mask, return_counts=True)
            organ_ids = organ_ids.long()
            organ_counts = organ_counts[organ_ids != 0]
            organ_ids = organ_ids[organ_ids != 0]

            # organs not touch boundary
            intact_organ_ids = [organ_id for organ_id, organ_count in zip(organ_ids, organ_counts) if organ_id not in boundary_organs]
            intact_organ_ids = torch.tensor(intact_organ_ids, device=masks.device).long()
            intact_organ_ids = intact_organ_ids - 1
            
            if not len(intact_organ_ids):
                continue

            organ_sizes = dict(zip([self.organs[organ_id] for organ_id in intact_organ_ids], [organ_counts[organ_ids == organ_id + 1].item() for organ_id in intact_organ_ids]))

            for organ_id in intact_organ_ids:
                organ_name = self.organs[organ_id.item()]
                if organ_name not in test_organs:
                    continue
                    
                if organ_name in organ_feat_dict:
                    continue

                tokens1 = organ_token_flags1[i, organ_id, :]
                tokens2 = organ_token_flags2[i, organ_id, :]
                tokens3 = organ_token_flags3[i, organ_id, :]

                query = self.query_tokens[organ_id].unsqueeze(0).unsqueeze(0)
                key1 = embed1[tokens1].unsqueeze(0)
                key2 = embed2[tokens2].unsqueeze(0)
                key3 = embed3[tokens3].unsqueeze(0)

                key = value = torch.cat([key1, key2, key3], dim=1)
                
                updated_query_token, _ = self.attention(query, key, value)
                updated_query_token = updated_query_token.squeeze(0)

                image_feat = F.normalize(self.vision_projs[organ_id](updated_query_token), dim=-1)
                
                organ_feat_dict[organ_name] = image_feat.cpu().tolist()

                for item in organ_logits.keys():
                    if isinstance(item, str):
                        item_organ_name = item.split('_')[0]
                    else:
                        item_organ_name = item[0]
                    if item_organ_name != organ_name:
                        continue

                    text_feat = text_feat_dict[item]

                    logits = image_feat @ text_feat.t() / self.temp
                    probs = logits.softmax(-1)
                    organ_logits[item].append(probs.cpu().tolist())
    
        return organ_logits, seg_probs

@torch.inference_mode()
def evaluate(pad_func, model, img_dir, save_dir, save_tag):

    datafolder = DataFolder(img_dir)
    dataloader = DataLoader(
        datafolder,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        drop_last=False,
        collate_fn=collate_fn
    )

    sw_batch_size = 1
    overlap = 0.25
    roi_size = (96, 256, 384)

    miss_num = 0
    results = []
    organ_status = {}

    # load pos/neg ensembled prompt embeddings
    text_feat_dict = torch.load('../ckpt/text_feat_dict_radar.pt')
    organ_feat_dict = {}
    save_path = os.path.join(save_dir, f'RADAR_infer_results_{save_tag}.csv')
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (image, test_items, meta_info) in enumerate(tqdm(dataloader, desc='Infer')):
        torch.cuda.empty_cache()
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

        test_organs = meta_info['test_organ_names']

        image_size = list(image.shape[2:])
        num_spatial_dims = len(image.shape) - 2

        scan_interval = _get_scan_interval(
            image_size, roi_size, num_spatial_dims, overlap
        )
        slices = dense_patch_slices(image_size, roi_size, scan_interval)
        num_win = len(slices)
        organ_logits = dict(zip(test_items, [[] for _ in test_items]))
        # organ_logits.pop('胆囊_术后胆囊缺失')  # surgically_absent_gallbladder

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

        # for melrin data, we just infer all organs
        # organ_logits = {k:v for k,v in organ_logits.items() if datafolder.organs.index(k[0]) in intact_organ_ids}

        for k, v in organ_logits.items():
            if not len(v):
                organ_name = k.split('_')[0]
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
  
        res = [meta_info['file_name']] + [''] * len(datafolder.test_items)
        organ_logits = {item: probs for item, probs in organ_logits.items() if len(probs) > 0}
        
        for item, probs in organ_logits.items():
            res[datafolder.test_items.index(item) + 1] = np.concatenate(probs).mean(0)[1]  # get average of one organ in multi-widows
        results.append(res)
    
    if dist.is_initialized():
        results = np.concatenate(all_gather(results), axis=0)
    else:
        results = results
    
    pd.DataFrame(
        results,
        columns=['file_name'] + [k for k in datafolder.test_items]
    ).to_csv(save_path, index=False, encoding='utf-8-sig')

def initialize():
    """
    Returns: transforms.DivisiblePadd, RADAR
    """
    print('\n--> Start initializing...')
    pad_func = transforms.DivisiblePadd(
        keys=["image", "label"],
        k=32,
        mode='constant',
        constant_values=0,
        method="end"
    )

    vision_encoder = VisionBranch()
    text_encoder = XBertEncoder.from_config({}, from_pretrained=True)

    model = RADAR(
        image_encoder=vision_encoder,
        text_encoder=text_encoder,
    )

    ckpt_path = os.path.join(model_root, "checkpoint_radar_pretrain.pth")
    print('--> ckpt_path: ', ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    msg = model.load_state_dict(ckpt['model'], strict=False)

    model.eval()
    model.cuda()

    print('\n--> Initialize done')

    return pad_func, model

def inference(initialize_returns, img_dir, save_dir, save_tag):
    """
    Args:
        initialize_returns: pad_func, model
        img_dir: see argparse
        save_dir: see argparse
    """
    print('\n--> Start inference.')
    pad_func, model = initialize_returns
    evaluate(pad_func, model, img_dir, save_dir, save_tag)
    csv_file = os.path.join(save_dir, f'RADAR_infer_results_{save_tag}.csv')
    print(f'evaluate done, save result_csv to {csv_file}.')
    
    # TODO: compute metrics
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='../data/demo_cases', help='The path to inference image folder.')
    parser.add_argument('--save_dir', type=str, default='../results', help='The path to save folder.')
    parser.add_argument('--save_tag', type=str, default='demo', help='Save tag.')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    initialize_returns = initialize()
    
    # infer
    inference(initialize_returns, args.img_dir, args.save_dir, args.save_tag)


if __name__ == '__main__':
    main()



