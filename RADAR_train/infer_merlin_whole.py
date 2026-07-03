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
    disease_name="coronary_calcification",
    region="heart",
    positive_prompts=[
                      "marked coronary calcification",
                      "severe coronary calcifications",
                      "coronary calcification",
                    ],
    negative_prompts=["heart are normal",
                      "heart is normal",
                      "heart appears normal",
                    ],
)
disease_prompts["coronary_calcification"] = prompts

prompts = DiseasePrompts(
    disease_name="thrombosis",
    region="vasculature",
    positive_prompts=[
                      "there is thrombosis",
                      "there has been interval thrombosis",
                      "partial thrombosis",
                      "portal vein thrombosis",
                      "complete thrombosis",
                      "stable thrombosis",
                      "likely represents thrombosis",
                      "in keeping with thrombosis",
                      "possible thrombosis",
                      "with thrombosis",
                      "chronic thrombosis",
                      "occlusive portal venous thrombus",
                      "occlusion of the portal vein with thrombus",
                      "nonocclusive thrombus",
                      "there is a thrombus",
                      "occlusive thrombus",
                    ],
    negative_prompts=["no evidence of deep vein thrombosis",
                      "no portal venous thrombosis",
                      "no thrombosis",
                      "no definite evidence of thrombosis",
                      "occlusive thrombosis",
                      "no evidence of venous thrombosis",
                      "no evidence of thrombosis",
                      "no venous thrombosis",
                      "without evidence of thrombosis",
                      "no evidence of ivc or pelvic vein thrombosis",
                      "no splenic , or portal vein thrombosis",
                      "negative for portal vein thrombus",
                      "no definite evidence for thrombus",
                      "no evidence of thrombus",
                      "no portal venous thrombus",
                      "no evidence of thrombus",
                      "no thrombus",
                      "no occlusion or thrombus",
                      "no venous thrombus",
                      ]
)
disease_prompts["thrombosis"] = prompts

prompts = DiseasePrompts(
    disease_name="metastatic_disease",
    region="multiple",
    positive_prompts=[
                      "consistent with metastatic disease",
                      "concerning for metastatic disease",
                      "may represent metastatic disease",
                      "suggestive of pleural metastatic disease",
                      "may reflect a metastatic lesion",
                      "consistent with worsening metastatic disease",
                      "concerning for nodal metastatic disease",
                      "lung bases concerning for metastatic disease",
                      "lesions consistent with metastatic disease",
                      "multiple intrahepatic metastatic lesions",
                      "compatible with metastatic pancreatic cancer",
                      "reflecting metastatic disease",
                      "concerning for worsening metastatic disease",
                      "likely secondary to peritoneal metastatic disease",
                      "compatible with metastatic rectal cancer",
                    ],
    negative_prompts=["likely benign",
                      "no evidence of metastatic disease",
                      "no definite evidence of metastatic disease",
                    ],
)
disease_prompts["metastatic_disease"] = prompts


prompts = DiseasePrompts(
    disease_name="osteopenia",
    region="musculoskeletal",
    positive_prompts=[
                      "diffuse osteopenia",
                      "marked osteopenia",
                      "mild osteopenia",
                      "patchy osteopenia",
                      "significant osteopenia",
                      "severe osteopenia",
                      "markedly osteopenic",
                      "bones are osteopenic",
                      "diffusely osteopenic",
                      "osteoporosis",
                      "osteoporotic",
                    ],
    negative_prompts=["musculoskeletal : normal ",
                    ],
)
disease_prompts["osteopenia"] = prompts


prompts = DiseasePrompts(
    disease_name="anasarca",
    region="abdominal wall",
    positive_prompts=["anasarca",
                      ],
    negative_prompts=["abdominal wall : normal .",
                    ],
)
disease_prompts["anasarca"] = prompts


prompts = DiseasePrompts(
    disease_name="lymphadenopathy",
    region="lymph nodes",
    positive_prompts=["lymphadenopathy is again seen",
                      "mesenteric lymphadenopathy",
                      "upper retroperitoneal lymphadenopathy",
                      "mild retroperitoneal lymphadenopathy",
                      "lower retroperitoneal lymphadenopathy",
                      "stable gastrohepatic ligament lymphadenopathy",
                      "stable gastrohepatic lymphadenopathy",
                      "stable lymphadenopathy",
                      "necrotic mesenteric lymphadenopathy",
                      "bilateral inguinal lymphadenopathy",
                      "right hilar lymphadenopathy",
                      "lymph nodes : extensive lymphadenopathy",
                      "aortic lymphadenopathy",
                      ],
    negative_prompts=["no lymphadenopathy"
                    ],
)
disease_prompts["lymphadenopathy"] = prompts

prompts = DiseasePrompts(
    disease_name="prostatomegaly",
    region="prostate and seminal vesicles",
    positive_prompts=["mild prostatomegaly",
                      "marked prostatomegaly",
                      "moderate prostatomegaly",
                      "prostate and seminal vesicles : prostatomegaly",
                      "nonspecific prostatomegaly",
                      "moderate non specific prostatomegaly",
                      "there is prostatomegaly",
                      "enlarged prostate .",
                      ],
    negative_prompts=["no prostatomegaly",
                      "prostate and seminal vesicles : normal ."
                    ],
)
disease_prompts["prostatomegaly"] = prompts


ascites_prompts = DiseasePrompts(
    disease_name="ascites",
    region="abdominal wall",
    positive_prompts=["medium volume ascites", "large volume ascites",
                      "moderate ascites",
                      "large ascites",
                      "trace ascites",
                      "small volume ascites",
                      "amount of ascites",
                      "volume of ascites",
                      "volume ascites",
                    ],
    negative_prompts=["no ascites", "no evidence of ascites",
                      "no amount of ascites",
                      "no volume of ascites",
                      "no volume ascites",
                    ],
)
disease_prompts["ascites"] = ascites_prompts



free_air_prompts = DiseasePrompts(
    disease_name="free air",
    region="abdominal wall",
    positive_prompts=[
        "there is free air",
        "foci of free air",
        "small amount of free air",
        "small focus of free air",
        "free air is seen",
        "a few locules of intraperitoneal free air",
        "with surrounding free air",
        "with intraperitoneal free air",
        "amount of intraperitoneal free air",
        "volume of free air",
        "is extensive associated free air",
        "is evidence of small intraperitoneal free air",
        "free air is also present",
        "with increased adjacent intra - abdominal free air",
        "volume intraperitoneal free air",
    ],
    negative_prompts=[
        "no free air",
        "or free air",
        "no evidence of free air",
        "no intraperitoneal free air",
        "there is no intra - peritoneal free air",
        "no evidence of extraluminal free air",
        "no extraluminal free air",
        "no intra - abdominal free air",
    ],
)
disease_prompts["free_air"] = free_air_prompts



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
            # keys=["image", "label"], 
            keys=["image"], 
            spatial_size=(96, 256, 384), 
            mode='constant', 
            constant_values=0,
            method="end"
        )

        self.test_items = []
        self.test_organs = []

        if dist.is_initialized():
            self.img_paths = self.img_paths[dist.get_rank()::dist.get_world_size()]

        self.pad_func = transforms.SpatialPadd(
            keys=["image"],
            spatial_size=(96, 256, 384), 
            mode='constant', 
            constant_values=0,
            method="end"
        )
        self.center_crop = transforms.CenterSpatialCropd(
            keys=["image"],
            roi_size=(96, 256, 384)
        )
        
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
        
        # reszie (to 5mm, 1, 1)
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
        
        # pad and center crop
        resized_data = self.pad_func(resized_data)
        resized_data = self.center_crop(resized_data)

        # clip, normalize to [0, 1], crop non-zero region, pad to [96, 256, 384]
        # Our default setting is [-300, 400], following merlin's setting, we used [-1000, 1000]
        img_resized = resized_data["image"]
        image = img_resized
        image[image>1000] = 1000
        image[image<-1000] = -1000
        image = (image - image.min()) / (image.max()- image.min())
        data["image"] = image
        
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

    miss_num = 0
    results = []
    organ_status = {}

    
    # --> get text feat
    num_positive_prompts = {}
    for disease in disease_prompts:
        disease_prompts[disease].positive_prompts = [prompt.replace("\\", "") for prompt in disease_prompts[disease].positive_prompts]
        disease_prompts[disease].negative_prompts = [prompt.replace("\\", "") for prompt in disease_prompts[disease].negative_prompts]
        num_positive_prompts[disease] = len(disease_prompts[disease].positive_prompts)

    with torch.no_grad():
        text_feat_dict = {}
        for disease,prompts in disease_prompts.items():
            text = model.tokenizer(
                prompts.positive_prompts + prompts.negative_prompts,
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
    save_path = os.path.join(ckp_dir, f'checkpoint_ZeroShotResult_whole.csv')
    
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
        organ_logits = dict(zip(test_items, [[] for _ in test_items]))
        
        # load whole layers
        seg_probs, seg, image_embeds1, image_embeds2, image_embeds3, organ_token_flags1, organ_token_flags2, organ_token_flags3 = model.visual_encoder(image, None)
        key_whole = value_whole = torch.cat([image_embeds1, image_embeds2, image_embeds3], dim=1)
        query_whole = model.query_tokens_whole.unsqueeze(0)
        query_whole = query_whole.expand(key_whole.shape[0], -1, -1)
        image_feat_whole, _ = model.attention_whole(query_whole, key_whole, value_whole)
        image_feat_whole = image_feat_whole.squeeze(1)
        image_feat_whole = model.vision_projs_whole(image_feat_whole)
        image_feat_whole = F.normalize(image_feat_whole, dim=-1)

        for k in organ_logits:
            text_feat = text_feat_dict[k]
            similarity = image_feat_whole @ text_feat.t() / model.temp
            
            similarity_positive = similarity[:, :num_positive_prompts[k]]
            similarity_positive = similarity_positive.mean(dim=1, keepdim=True)
            
            similarity_negative = similarity[:, num_positive_prompts[k]:]
            similarity_negative = similarity_negative.mean(dim=1, keepdim=True)
            similarity = torch.cat([similarity_negative, similarity_positive], dim=1)
        
            probs = similarity.softmax(-1)

            organ_logits[k].append(probs[0][-1].detach().cpu().numpy())

        res = [meta_info['file_name']] + [''] * len(test_items)
        organ_logits = {item: probs for item, probs in organ_logits.items() if len(probs) > 0}
        for item, probs in organ_logits.items():
            res[test_items.index(item) + 1] = probs[0].tolist()
        results.append(res)
    
    if dist.is_initialized():
        results = np.concatenate(all_gather(results), axis=0)
        organ_feat_dict = all_gather(organ_feat_dict)
    else:
        organ_feat_dict = [organ_feat_dict]
    
    if rank == 0:
        pd.DataFrame(
            results,
            columns=['file_name'] + [k for k in test_items]
        ).to_csv(save_path, index=False, encoding='utf-8')
        
        print('Save result file successfully!')

if __name__ == '__main__':
    evaluate()
 
    if dist.is_initialized():  
        dist.destroy_process_group()
    
    torch.cuda.empty_cache()
