"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see THIRD_PARTY_LICENSES.md or https://opensource.org/licenses/BSD-3-Clause
"""

from copy import deepcopy
import re
import torch
import torch.distributed as dist

from lavis.common.registry import registry
from lavis.common.dist_utils import is_dist_avail_and_initialized
from lavis.models.radar_models import tie_encoder_decoder_weights
from lavis.models.radar_models.radar_base import RadarBase
from lavis.models.radar_models.dice import MemoryEfficientSoftDiceLoss
from lavis.models.radar_models.radar_outputs import RadarOutput
from lavis.models.base_model import (
    MomentumDistilationMixin,
    SharedQueueMixin,
    all_gather_with_grad,
    concat_all_gather
)
from lavis.models.med import XBertEncoder, XBertLMHeadDecoder
from torch import nn
import random
import os
import json
import numpy as np
import torch.nn.functional as F
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import wordpunct_tokenize
import torch
from lavis.models.radar_models.vision_branch import VisionBranch


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

@registry.register_model("radar_pretrain")
class RadarPretrain(RadarBase, SharedQueueMixin, MomentumDistilationMixin):
    """
    RADAR pretrain model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": ""
    }

    def __init__(
        self,
        image_encoder,
        text_encoder,
        text_decoder,
        queue_size,
        alpha=0.4,
        embed_dim=256,
        momentum=0.995,
        tie_enc_dec_weights=False,
        max_txt_len=512,
        radar_plus=True
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.text_encoder = text_encoder

        self.visual_encoder = image_encoder

        # creating projection layers for ITC
        text_width = 768
        vision_width = 256

        self.text_proj = nn.Linear(text_width, embed_dim)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.alpha = alpha
        self.max_txt_len = max_txt_len
        self.radar_plus = radar_plus

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
        
        self.attention = nn.MultiheadAttention(
            embed_dim=vision_width,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.vision_projs = nn.ModuleList([nn.Linear(vision_width, embed_dim) for _ in range(len(self.organs))])
        self.query_tokens = nn.Parameter(torch.zeros(len(self.organs), vision_width))
        
        self.dice_loss = MemoryEfficientSoftDiceLoss(apply_nonlin=None, batch_dice=False, do_bg=False, smooth=0, ddp=False)
        
        # define whole image contrastive learning
        self.vision_projs_whole = nn.Linear(vision_width, embed_dim)
        self.attention_whole = nn.MultiheadAttention(
            embed_dim=vision_width,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.query_tokens_whole = nn.Parameter(torch.zeros(1, vision_width))
        
        self.text_encoder_m = deepcopy(self.text_encoder)
        self.text_proj_m = deepcopy(self.text_proj)
        self.model_pairs = [
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()
        

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    def forward(self, samples):
        image = samples["image"]
        seg_label = samples["seg"]
        organ_captions = samples["text_input"]
        index = samples["index"]
        patient_id  = samples["patient_id"]
        organ_abnormal_flags = samples["organ_abnormal_flags"]
        
        whole_report = organ_captions['report']
        
        # image embeddings and features
        seg_probs, seg, image_embeds1, image_embeds2, image_embeds3, organ_token_flags1, organ_token_flags2, organ_token_flags3 = self.visual_encoder(image, None)
        
        cur_iters = samples['iters']
        if cur_iters % 2 == 0 and self.radar_plus:
            # --> whole image contrastive learning
            key_whole = value_whole = torch.cat([image_embeds1, image_embeds2, image_embeds3], dim=1)
            query_whole = self.query_tokens_whole.unsqueeze(0)
            query_whole = query_whole.expand(key_whole.shape[0], -1, -1)
            image_feat_whole, _ = self.attention_whole(query_whole, key_whole, value_whole)
            image_feat_whole = image_feat_whole.squeeze(1)
            
            image_feat_whole = self.vision_projs_whole(image_feat_whole)
            image_feat_whole = F.normalize(image_feat_whole, dim=-1)

            cl_text_input_whole = [textl for textl in whole_report]
            
            text_whole = self.tokenizer(
                cl_text_input_whole,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(image.device)
            text_output_whole = self.text_encoder.forward_text(text_whole)
            text_embeds_whole = text_output_whole.last_hidden_state
            text_feat_whole = F.normalize(self.text_proj(text_embeds_whole[:, 0, :]), dim=-1)

            # NOTE: gather image and text feats
            if is_dist_avail_and_initialized():
                image_feat_all_whole = [feat.to(image_feat_whole.device) for feat in all_gather(image_feat_whole)]
                text_feat_all_whole = [feat.to(text_feat_whole.device) for feat in all_gather(text_feat_whole)]

                image_feat_all_whole[dist.get_rank()] = image_feat_whole
                text_feat_all_whole[dist.get_rank()] = text_feat_whole

                image_feat_all_whole = torch.cat(image_feat_all_whole, dim=0)
                text_feat_all_whole = torch.cat(text_feat_all_whole, dim=0)

            else:
                image_feat_all_whole = image_feat_whole
                text_feat_all_whole = text_feat_whole

            sim_i2t_whole = image_feat_all_whole @ text_feat_all_whole.t() / self.temp
            sim_t2i_whole = text_feat_all_whole @ image_feat_all_whole.t() / self.temp
                
            with torch.no_grad():
                sim_targets_whole = torch.zeros(sim_i2t_whole.size()).to(image.device)
                sim_targets_whole.fill_diagonal_(1)
                
            sim_i2t_targets_whole = sim_targets_whole
            sim_t2i_targets_whole = sim_targets_whole

            loss_i2t_whole = - torch.sum(
                F.log_softmax(sim_i2t_whole, dim=1) * sim_i2t_targets_whole, dim=1
            ).mean()
            
            loss_t2i_whole = - torch.sum(
                F.log_softmax(sim_t2i_whole, dim=1) * sim_t2i_targets_whole, dim=1
            ).mean()
            
            loss_itc = (loss_i2t_whole + loss_t2i_whole) / 2
            organ_wise_loss_itc = {}
        else:
            # --> anatomy-wise contrastive learning
            with torch.no_grad():
                organ_intact_flags = torch.zeros(len(seg), len(self.organs), dtype=bool, device=seg.device)
                for i, pul_seg in enumerate(seg):
                    boundaries = [
                        pul_seg[0], pul_seg[-1],
                        pul_seg[:, 0], pul_seg[:, -1],
                        pul_seg[:, :, 0], pul_seg[:, :, -1]
                    ]
                    
                    non_zero_boundaries = [b[b != 0].flatten() for b in boundaries]
                    boundary_values = torch.cat(non_zero_boundaries)
                    boundary_organs = torch.unique(boundary_values)

                    organ_ids, organ_counts = torch.unique(pul_seg, return_counts=True)
                    organ_ids = organ_ids[organ_ids > 0]
                    
                    intact_organ_ids = [organ_id for organ_id in organ_ids if organ_id not in boundary_organs]
                    intact_organ_ids = torch.tensor(intact_organ_ids).long()
                        
                    organ_intact_flags[i][intact_organ_ids - 1] = True    # [12, 36]

            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)

            # criteria to calculate loss
            with torch.no_grad():
                organ_status_world = (organ_abnormal_flags & organ_intact_flags).sum(0)
                if is_dist_avail_and_initialized():
                    dist.all_reduce(organ_status_world, op=dist.ReduceOp.SUM)
            
            cl_organ_ids = torch.where(organ_status_world)[0]
            
            organ_wise_loss_itc = {}
            for cl_organ_id in cl_organ_ids:
                organ_name = self.organs[cl_organ_id]

                cl_patient_ids = torch.where(organ_intact_flags[:, cl_organ_id])[0]
                
                # --> Find cl_organ_id and cl_patient_ids for normal/abnormal flags
                batch_organ_abnormal_flags = organ_abnormal_flags[cl_patient_ids, cl_organ_id]

                if not len(cl_patient_ids):
                    image_feat = torch.empty(0, 256, dtype=torch.float).to(image.device)
                    text_feat = torch.empty(0, 256, dtype=torch.float).to(image.device)
                    text_feat_m = torch.empty(0, 256, dtype=torch.float).to(image.device)
                    cl_text_input = []
                else:
                    image_feat = self.get_roi_features(image_embeds1, image_embeds2, image_embeds3, organ_token_flags1, organ_token_flags2, organ_token_flags3, cl_patient_ids, cl_organ_id)
                    image_feat = self.vision_projs[cl_organ_id](image_feat)
                    image_feat = F.normalize(image_feat, dim=-1)

                    cl_text_input = [organ_captions[organ_name][cl_patient_id] for cl_patient_id in cl_patient_ids]
                    cl_text_input = [textl for textl in cl_text_input]
                    dynamic_txt_len = self.max_txt_len

                    text = self.tokenizer(
                        cl_text_input,
                        padding="max_length",
                        truncation=True,
                        max_length=dynamic_txt_len,
                        return_tensors="pt",
                    ).to(image.device)
                    text_output = self.text_encoder.forward_text(text)
                    text_embeds = text_output.last_hidden_state
                    text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
                    with torch.no_grad():
                        self._momentum_update()
                        text_output_m = self.text_encoder_m.forward_text(text)
                        text_embeds_m = text_output_m.last_hidden_state
                        text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
                # NOTE: gather image and text feats
                if is_dist_avail_and_initialized():
                    image_feat_all = [feat.to(image_feat.device) for feat in all_gather(image_feat)]
                    text_feat_all = [feat.to(text_feat.device) for feat in all_gather(text_feat)]
                    text_feat_all_m = [feat.to(text_feat_m.device) for feat in all_gather(text_feat_m)]
                    

                    image_feat_all[dist.get_rank()] = image_feat
                    text_feat_all[dist.get_rank()] = text_feat
                    text_feat_all_m[dist.get_rank()] = text_feat_m
                    

                    image_feat_all = torch.cat(image_feat_all, dim=0)
                    text_feat_all = torch.cat(text_feat_all, dim=0)
                    text_feat_all_m = torch.cat(text_feat_all_m, dim=0)

                else:
                    image_feat_all = image_feat
                    text_feat_all = text_feat
                    text_feat_all_m = text_feat_m

                cl_text_input = np.array(cl_text_input)
                batch_organ_abnormal_flags = np.array(batch_organ_abnormal_flags.cpu())
                if is_dist_avail_and_initialized():
                    gathered_cl_text_input = all_gather(cl_text_input)
                    cl_text_input_all = np.concatenate(gathered_cl_text_input)
                    gathered_batch_organ_abnormal_flags = all_gather(batch_organ_abnormal_flags)
                    batch_organ_abnormal_flags_all = np.concatenate(gathered_batch_organ_abnormal_flags)
                else:
                    cl_text_input_all = cl_text_input
                    batch_organ_abnormal_flags_all = batch_organ_abnormal_flags

                sim_i2t = image_feat_all @ text_feat_all.t() / self.temp
                sim_t2i = text_feat_all @ image_feat_all.t() / self.temp
                
                with torch.no_grad():
                    sim_targets = torch.zeros(sim_i2t.size()).to(image.device)
                    sim_targets.fill_diagonal_(1)
                    
                    normal_flag = [not abflag for abflag in batch_organ_abnormal_flags_all]
                    normal_flag = np.array(normal_flag)
                    abnormal_flag = [abflag for abflag in batch_organ_abnormal_flags_all] 
                    abnormal_flag = np.array(abnormal_flag)

                    semantic_matrix_batch = normal_flag[:, None] * normal_flag[None, :]
                    semantic_matrix_batch = semantic_matrix_batch.astype(float)
                    abnormal_flag_matrix = abnormal_flag[:, None] * abnormal_flag[None, :]
                    abnormal_flag_matrix = abnormal_flag_matrix.astype(float)
                    
                    semantic_matrix_batch1 = cl_text_input_all[:, None] == cl_text_input_all[None, :]
                    semantic_matrix_batch1 = semantic_matrix_batch1.astype(float)

                    semantic_matrix_batch = semantic_matrix_batch + semantic_matrix_batch1
                    semantic_matrix_batch = semantic_matrix_batch.astype(bool)
                    
                    # 
                    sim_t2t_m = text_feat_all_m @ text_feat_all_m.t() / self.temp
                    semantic_matrix_batch_abnormal = abnormal_flag_matrix * (F.softmax(sim_t2t_m, dim=1)).cpu().numpy()
                    
                    # 
                    semantic_matrix_batch = semantic_matrix_batch.astype(float) + semantic_matrix_batch_abnormal
                    
                    semantic_matrix_batch = torch.from_numpy(semantic_matrix_batch).to(image.device)
                    semantic_matrix_batch.fill_diagonal_(0)

                    sim_targets += semantic_matrix_batch
                    sim_targets /= sim_targets.sum(1, keepdim=True)

                if len(torch.unique(sim_targets)) == 1 or not len(cl_text_input):
                    continue
                
                sim_i2t_targets = sim_targets
                sim_t2i_targets = sim_targets

                loss_i2t = - torch.sum(
                    F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1
                ).mean()
                
                loss_t2i = - torch.sum(
                    F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1
                ).mean()
                
                loss_itc = (loss_i2t + loss_t2i) / 2
                organ_wise_loss_itc.update({f'{organ_name}_itc': loss_itc})
            loss_itc = sum(organ_wise_loss_itc.values())
        
        # compute seg loss
        target_size = seg_probs.shape[-3:]
        seg_label = F.interpolate(seg_label.unsqueeze(1), size=target_size, mode='nearest')
        loss_seg = self.dice_loss(seg_probs, seg_label)
 
        return RadarOutput(
            loss=loss_itc+loss_seg,
            loss_seg=loss_seg,
            loss_itc=loss_itc,
            organ_wise_loss_itc=organ_wise_loss_itc
        )
    
    def get_roi_features(self, image_embeds1, image_embeds2, image_embeds3, organ_token_flags1, organ_token_flags2, organ_token_flags3, cl_patient_ids, cl_organ_id):
        query = self.query_tokens[cl_organ_id].unsqueeze(0).unsqueeze(0)

        roi_feats = []
        for image_embed1, image_embed2, image_embed3, tokens1, tokens2, tokens3 in zip(image_embeds1[cl_patient_ids], image_embeds2[cl_patient_ids], image_embeds3[cl_patient_ids], organ_token_flags1[cl_patient_ids, cl_organ_id], organ_token_flags2[cl_patient_ids, cl_organ_id], organ_token_flags3[cl_patient_ids, cl_organ_id]):
            key1 = image_embed1[tokens1].unsqueeze(0)
            key2 = image_embed2[tokens2].unsqueeze(0)
            key3 = image_embed3[tokens3].unsqueeze(0)

            key = value = torch.cat([key1, key2, key3], dim=1)
            
            updated_query_token, _ = self.attention(query, key, value)
            roi_feats.append(updated_query_token.squeeze(0))

        roi_feats = torch.cat(roi_feats, dim=0)
        return roi_feats

    @classmethod
    def from_config(cls, cfg=None):

        image_encoder = VisionBranch()
        text_encoder = XBertEncoder.from_config(cfg, from_pretrained=True)
        text_decoder = None

        embed_dim = cfg.get("embed_dim", 256)
        momentum = cfg.get("momentum", 0.995)
        alpha = cfg.get("alpha", 0.4)
        max_txt_len = cfg.get("max_txt_len", 512)
        queue_size = cfg.get("queue_size", 57600)
        radar_plus = cfg.get("radar_plus", True)
        radar_ft = cfg.get("radar_ft", True)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            text_decoder=text_decoder,
            embed_dim=embed_dim,
            queue_size=queue_size,
            momentum=momentum,
            alpha=alpha,
            tie_enc_dec_weights=False,
            max_txt_len=max_txt_len,
            radar_plus=radar_plus
        )

        model.load_checkpoint_from_config(cfg)
        
        if radar_ft:
            # load RADAR pretrained ckpt
            ckpt_path = '../ckpt/checkpoint_radar_pretrain.pth'
            ckpt = torch.load(ckpt_path, map_location='cpu')

            from collections import OrderedDict
            new_ckpt = OrderedDict()
            for key, value in ckpt['model'].items():
                if key.startswith("text_"):
                    continue
                else:
                    new_ckpt[key] = value
            # init whole with anatomy weight
            new_ckpt['query_tokens_whole'] = ckpt['model']['query_tokens'][0:1, :]
            new_ckpt['vision_projs_whole.weight'] = ckpt['model']['vision_projs.0.weight']
            new_ckpt['vision_projs_whole.bias'] = ckpt['model']['vision_projs.0.bias']
            new_ckpt['attention_whole.in_proj_weight'] = ckpt['model']['attention.in_proj_weight']
            new_ckpt['attention_whole.in_proj_bias'] = ckpt['model']['attention.in_proj_bias']
            new_ckpt['attention_whole.out_proj.weight'] = ckpt['model']['attention.out_proj.weight']
            new_ckpt['attention_whole.out_proj.bias'] = ckpt['model']['attention.out_proj.bias']
            msg = model.load_state_dict(new_ckpt, strict=False)
            print('\n\n --> load pre-trained RADAR ckpt in vision branch, do not load weight of text encoder. \n')

        return model
    
    def sanitize_report(self, report):
        report = report.lower()
        return " ".join(wordpunct_tokenize(report))

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum
                )

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
                    if item[0] != organ_name:
                        continue

                    text_feat = text_feat_dict[item]
                    
                    if text_feat.shape[0] > 2:  # prompt ensemble
                        similarity = image_feat @ text_feat.t() / self.temp
                
                        similarity_positive = similarity[:, :self.num_positive_prompts[item]]
                        similarity_positive = similarity_positive.mean(dim=1, keepdim=True)
                        
                        similarity_negative = similarity[:, self.num_positive_prompts[item]:]
                        similarity_negative = similarity_negative.mean(dim=1, keepdim=True)
                        similarity = torch.cat([similarity_negative, similarity_positive], dim=1)
                    
                        probs = similarity.softmax(-1)
                        organ_logits[item].append(probs.detach().cpu().numpy().tolist())
                    else:
                        logits = image_feat @ text_feat.t() / self.temp
                        probs = logits.softmax(-1)
                        organ_logits[item].append(probs.cpu().tolist())
    
        return organ_logits, seg_probs
 
    def prepare_text_feat(self, test_items, length=None):
        if length is None:
            length = self.max_txt_len

        device = self.text_encoder.device
        text_feat_dict = {}
        for prompt, item in zip(*self._get_prompt(test_items)):
            text = self.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=length,
                return_tensors="pt",
            ).to(device)

            text_output = self.text_encoder.forward_text(text)
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
            text_feat_dict[tuple(item)] = text_feat

        return text_feat_dict
    
    # @staticmethod
    def _get_prompt(
        self,
        test_items,
        organ_name: str = None
    ) -> str:
        if organ_name is not None:
            test_items = [item for item in test_items if item[0] == organ_name]

        negative_prompts = [item[2] for item in test_items]
        positive_prompts = [item[3] for item in test_items]

        prompts = list(zip(negative_prompts, positive_prompts))

        return prompts, test_items
