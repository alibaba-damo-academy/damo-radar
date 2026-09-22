"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see THIRD_PARTY_LICENSES.md or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.processors.base_processor import BaseProcessor
from lavis.processors.radar_processors import RadarReportProcessor, RadarImageTrainProcessor
from lavis.common.registry import registry

__all__ = [
    "BaseProcessor",
    "RadarReportProcessor",
    "RadarImageTrainProcessor"
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
