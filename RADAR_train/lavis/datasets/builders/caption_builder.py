"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see THIRD_PARTY_LICENSES.md or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.radar_datasets import RadarDataset

from lavis.common.registry import registry


@registry.register_builder("radar_dataset")
class RadarBuilder(BaseDatasetBuilder):
    train_dataset_cls = RadarDataset

    DATASET_CONFIG_DICT = {
        "default": "",
    }
