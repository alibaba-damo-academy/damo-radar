
from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn as nn
import sys

from monai.utils import deprecated_arg
import pydoc
import warnings
from typing import Union
import torch.nn.functional as F

__all__ = ["VisionBranch"]


class VisionBranch(nn.Module):
    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels=1
    ) -> None:
        super().__init__()
        
        # define UNet
        self.UNet = self.get_network_from_plans(
            arch_class_name="dynamic_network_architectures.architectures.unet_lightdecoder.PlainConvUNetLightD",
            arch_kwargs={
                "n_stages": 6,
                "features_per_stage": [32, 64, 128, 256, 320, 320],
                "conv_op": "torch.nn.modules.conv.Conv3d",
                "kernel_sizes": [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "strides": [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
                "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
                "conv_bias": True,
                "norm_op": "torch.nn.BatchNorm3d",
                "norm_op_kwargs": {},
                "dropout_op": None,
                "dropout_op_kwargs": None,
                "nonlin": "torch.nn.ReLU",
                "nonlin_kwargs": {"inplace": True},
            },
            arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
            input_channels=1,
            output_channels=37,
            allow_init=True,
            deep_supervision=True,
        )
        
        # load pretrained nnu checkpoint
        ckpt_path = '../ckpt/checkpoint_unet.pth'
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        network_weights = checkpoint["network_weights"]
        msg = self.UNet.load_state_dict(network_weights, strict=False)
        
        self.proj1 = nn.Conv3d(320, 256, kernel_size=1)
        self.proj2 = nn.Conv3d(320, 256, kernel_size=1)
        self.proj3 = nn.Conv3d(256, 256, kernel_size=1)
        
        self.organs = [
            '肾上腺', '主动脉', '竖脊肌', '脑', '锁骨', '大肠', '十二指肠', '食管', '面部', '股骨', 
            '胆囊', "臀肌", '心脏', '髋关节', '肱骨', '髂动脉', '髂静脉', '髂腰肌', '下腔静脉', '肾', 
            '肝', '肺', '胰腺', '门静脉', '肺动脉', '肋骨', '骶骨', '肩胛骨', '小肠', '脾', 
            '胃', '气管', '膀胱', '颈椎', '腰椎', '胸椎'
        ]  # only for length
        # organ_dict = {
        #     "肾上腺": "adrenal gland", "主动脉": "aorta", "竖脊肌": "erector spinae muscle", "脑": "brain", "锁骨": "clavicle", "大肠": "large bowel", "十二指肠": "duodenum", 
        #     "食管": "esophagus", "面部": "face", "股骨": "femur", "胆囊": "gallbladder", "臀肌": "gluteus muscle", "心脏": "heart", "髋关节": "hip joint", "肱骨": "humerus", 
        #     "髂动脉": "iliac artery", "髂静脉": "iliac vena", "髂腰肌": "iliopsoas muscle", "下腔静脉": "inferior vena cava", "肾": "kidney", "肝": "liver", "肺": "lung", 
        #     "胰腺": "pancreas", "门静脉": "portal vein", "肺动脉": "pulmonary artery", "肋骨": "rib", "骶骨": "sacrum", "肩胛骨": "scapula", "小肠": "small bowel", "脾": "spleen", 
        #     "胃": "stomach", "气管": "trachea", "膀胱": "bladder", "颈椎": "cervical vertebrae", "腰椎": "lumbar vertebrae", "胸椎": "thoracic vertebrae"
        # }

    def get_network_from_plans(sefl, arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
        network_class = arch_class_name
        architecture_kwargs = dict(**arch_kwargs)
        for ri in arch_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        nw_class = pydoc.locate(network_class)

        if deep_supervision is not None:
            architecture_kwargs['deep_supervision'] = deep_supervision

        network = nw_class(
            input_channels=input_channels,
            num_classes=output_channels,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize') and allow_init:
            network.apply(network.initialize)

        return network

    def forward(self, x, y):
        skips, segs = self.UNet(x)
        
        scale1 = self.proj1(skips[-1])
        scale2 = self.proj2(skips[-2])
        scale3 = self.proj3(skips[-3])
        
        # process mask
        pred_logit = segs[0]
        seg_probs = torch.softmax(pred_logit, 1)
        target_size = [pred_logit.shape[-3], pred_logit.shape[-2]*2, pred_logit.shape[-1]*2]
        pred_logit = F.interpolate(pred_logit, size=target_size, mode='trilinear', align_corners=False)
        pred_mask = torch.softmax(pred_logit, 1)
        pred_mask = pred_mask.argmax(1)
        y = pred_mask
        
        res_x1 = scale1.flatten(2).transpose(1, 2)
        res_x2 = scale2.flatten(2).transpose(1, 2)
        res_x3 = scale3.flatten(2).transpose(1, 2)
        
        B, L1, _ = res_x1.size()
        B, L2, _ = res_x2.size()
        B, L3, _ = res_x3.size()
        
        with torch.no_grad():
            organ_token_flags1 = torch.zeros(B, len(self.organs), L1, dtype=bool).to(x.device)
            organ_token_flags2 = torch.zeros(B, len(self.organs), L2, dtype=bool).to(x.device)
            organ_token_flags3 = torch.zeros(B, len(self.organs), L3, dtype=bool).to(x.device)

            b = x.size(0)
            for i in range(b):
                unique_values = torch.unique(y[i])
                unique_values = unique_values[unique_values != 0]
                if unique_values.tolist() == []:
                    continue
                masks = torch.stack([torch.eq(y[i], uv) for uv in unique_values]).float()
                
                highlight_tokens3 = F.max_pool3d(
                    masks.unsqueeze(1),
                    kernel_size=(2, 8, 8),
                    stride=(2, 8, 8)
                ).flatten(1) > 0

                highlight_tokens2 = F.max_pool3d(
                    masks.unsqueeze(1),
                    kernel_size=(4, 16, 16),
                    stride=(4, 16, 16)
                ).flatten(1) > 0

                highlight_tokens1 = F.max_pool3d(
                    masks.unsqueeze(1),
                    kernel_size=(8, 32, 32),
                    stride=(8, 32, 32)
                ).flatten(1) > 0

                organ_token_flags1[i][unique_values.long() - 1] = highlight_tokens1 > 0
                organ_token_flags2[i][unique_values.long() - 1] = highlight_tokens2 > 0
                organ_token_flags3[i][unique_values.long() - 1] = highlight_tokens3 > 0
        return seg_probs, pred_mask, res_x1, res_x2, res_x3, organ_token_flags1, organ_token_flags2, organ_token_flags3

