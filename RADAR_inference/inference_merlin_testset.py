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
        
        if not os.path.exists(img_dir):
            print('Please modify the --img_dir to your own path.')
            assert False
        
        merlin_info = json.load(open('../ckpt/merlin_report.json'))
        patient_list = []
        for id,minfo in merlin_info.items():
            if minfo['split'] == 'test':
                patient_list.append(id+'.nii.gz')

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
        
        self.test_items = ['主动脉_主动脉瘤', '主动脉_粥样硬化', '大肠_粘膜下水肿', '大肠_阑尾炎', '小肠_梗阻', '心脏_主动脉瓣钙化', '心脏_心影（脏）增大', '肝_肝内胆管扩张', '肝_肝大', '肝_脂肪肝', '肺_胸腔积液', '肺_膨胀不全', '肾_低密度影', '肾_囊肿', '肾_肾积水', '胆囊_结石', '胰腺_萎缩', '脾_脾大', '腰椎_骨折', '食管_裂孔疝', '胆囊_术后胆囊缺失']
        self.map_radar_merlin = {'主动脉_主动脉瘤': 'abdominal_aortic_aneurysm', '主动脉_粥样硬化': 'atherosclerosis', '大肠_粘膜下水肿': 'submucosal_edema', '大肠_阑尾炎': 'appendicitis', '小肠_梗阻': 'bowel_obstruction', '心脏_主动脉瓣钙化': 'aortic_valve_calcification', '心脏_心影（脏）增大': 'cardiomegaly', '肝_肝内胆管扩张': 'biliary_ductal_dilation', '肝_肝大': 'hepatomegaly', '肝_脂肪肝': 'hepatic_steatosis', '肺_胸腔积液': 'pleural_effusion', '肺_膨胀不全': 'atelectasis', '肾_低密度影': 'renal_hypodensities', '肾_囊肿': 'renal_cyst', '肾_肾积水': 'hydronephrosis', '胆囊_结石': 'gallstones', '胰腺_萎缩': 'pancreatic_atrophy', '脾_脾大': 'splenomegaly', '腰椎_骨折': 'fracture', '食管_裂孔疝': 'hiatal_hernia', '胆囊_术后胆囊缺失': 'surgically_absent_gallbladder'}
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
    text_feat_dict = torch.load('../ckpt/infer_text_embedding_merlin.pt')
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
        organ_logits.pop('胆囊_术后胆囊缺失')  # surgically_absent_gallbladder

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

        # for melrin, we just infer all organs
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
        # pred "surgically_absent_gallbladder" using segmentation
        # gallbladder organ indice is 11
        if 11 not in torch.unique(stitched_mask):
            sag_prob = 0.
        else:
            sag_voxel_num  = (stitched_mask == 11).sum().item()
            sag_prob = sag_voxel_num    
        res = [meta_info['file_name']] + [''] * len(datafolder.test_items)
        organ_logits = {item: probs for item, probs in organ_logits.items() if len(probs) > 0}
        
        for item, probs in organ_logits.items():
            res[datafolder.test_items.index(item) + 1] = np.concatenate(probs).mean(0)[1]  # get average of one organ in multi-widows
        res[-1] = sag_prob  # add: surgically_absent_gallbladder
        results.append(res)
    
    if dist.is_initialized():
        results = np.concatenate(all_gather(results), axis=0)
    else:
        results = results
    
    pd.DataFrame(
        results,
        columns=['file_name'] + [f'{k}_({datafolder.map_radar_merlin[k]})' for k in datafolder.test_items]
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
    parser.add_argument('--img_dir', type=str, default=None, required=True, help='The path to inference image folder.')
    parser.add_argument('--save_dir', type=str, default='../results', help='The path to save folder.')
    parser.add_argument('--save_tag', type=str, default='MerlinTestset', help='Save tag.')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    initialize_returns = initialize()
    
    # infer
    inference(initialize_returns, args.img_dir, args.save_dir, args.save_tag)


if __name__ == '__main__':
    main()



