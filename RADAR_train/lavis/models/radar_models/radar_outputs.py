"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see THIRD_PARTY_LICENSES.md or https://opensource.org/licenses/BSD-3-Clause
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)

@dataclass
class RadarOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_itc: Optional[torch.FloatTensor] = None
    loss_itc_whole: Optional[torch.FloatTensor] = None
    loss_seg: Optional[torch.FloatTensor] = None

    organ_wise_loss_itm: Optional[torch.FloatTensor] = None
    organ_wise_loss_itc: Optional[torch.FloatTensor] = None
    organ_wise_loss_ce: Optional[torch.FloatTensor] = None
    organ_wise_loss_tri: Optional[torch.FloatTensor] = None
    organ_wise_loss_con: Optional[torch.FloatTensor] = None

    loss_mlm: Optional[torch.FloatTensor] = None
