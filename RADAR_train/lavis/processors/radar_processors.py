"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see THIRD_PARTY_LICENSES.md or https://opensource.org/licenses/BSD-3-Clause
"""

import re
from monai import transforms

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Normalize

class ImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        pass

@registry.register_processor("radar_report_processor")
class RadarReportProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=512):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 512)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, captions):
        if 'conc' in captions:
            del captions['conc']
        if 'desc' in captions:
            del captions['desc']

        for (organ, caption) in captions.items():
            caption = caption.lower()
            caption = re.sub(
                r"\s{2,}",
                " ",
                caption,
            )
            caption = caption.rstrip("\n")
            caption = caption.strip(" ")

            if caption[-1] != '.':
                caption += '.'

            captions[organ] = caption

        return captions


@registry.register_processor("radar_image_processor")
class RadarImageTrainProcessor(ImageBaseProcessor):
    def __init__(
        self, image_size=384, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose([
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.ToTensord(keys=["image", "label"])
        ])
    
    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("ifmage_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )
