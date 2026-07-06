import concurrent.futures
import multiprocessing
import os
import pathlib
import time
from functools import partial
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from monai import transforms
from tqdm import tqdm

root_dir = '/Merlin/download/merlinabdominalctdataset/merlin_data'  # melrin original data

def fuse_mask(mask_path):
    # print('mask_path', mask_path)
    try:
        # Step 1: Merge class
        relative_path = "/".join(mask_path.split("/")[-1:])
        Path(os.path.join(root_dir, "merged_masks/" + relative_path)).parent.mkdir(parents=True, exist_ok=True)

        class_map = {
            1: "spleen",
            2: "kidney_right",
            3: "kidney_left",
            4: "gallbladder",
            5: "liver",
            6: "stomach",
            7: "aorta",
            8: "inferior_vena_cava",
            9: "portal_vein_and_splenic_vein",
            10: "pancreas",
            11: "adrenal_gland_right",
            12: "adrenal_gland_left",
            13: "lung_upper_lobe_left",
            14: "lung_lower_lobe_left",
            15: "lung_upper_lobe_right",
            16: "lung_middle_lobe_right",
            17: "lung_lower_lobe_right",
            18: "vertebrae_L5",
            19: "vertebrae_L4",
            20: "vertebrae_L3",
            21: "vertebrae_L2",
            22: "vertebrae_L1",
            23: "vertebrae_T12",
            24: "vertebrae_T11",
            25: "vertebrae_T10",
            26: "vertebrae_T9",
            27: "vertebrae_T8",
            28: "vertebrae_T7",
            29: "vertebrae_T6",
            30: "vertebrae_T5",
            31: "vertebrae_T4",
            32: "vertebrae_T3",
            33: "vertebrae_T2",
            34: "vertebrae_T1",
            35: "vertebrae_C7",
            36: "vertebrae_C6",
            37: "vertebrae_C5",
            38: "vertebrae_C4",
            39: "vertebrae_C3",
            40: "vertebrae_C2",
            41: "vertebrae_C1",
            42: "esophagus",
            43: "trachea",
            44: "heart_myocardium",
            45: "heart_atrium_left",
            46: "heart_ventricle_left",
            47: "heart_atrium_right",
            48: "heart_ventricle_right",
            49: "pulmonary_artery",
            50: "brain",
            51: "iliac_artery_left",
            52: "iliac_artery_right",
            53: "iliac_vena_left",
            54: "iliac_vena_right",
            55: "small_bowel",
            56: "duodenum",
            57: "colon",
            58: "rib_left_1",
            59: "rib_left_2",
            60: "rib_left_3",
            61: "rib_left_4",
            62: "rib_left_5",
            63: "rib_left_6",
            64: "rib_left_7",
            65: "rib_left_8",
            66: "rib_left_9",
            67: "rib_left_10",
            68: "rib_left_11",
            69: "rib_left_12",
            70: "rib_right_1",
            71: "rib_right_2",
            72: "rib_right_3",
            73: "rib_right_4",
            74: "rib_right_5",
            75: "rib_right_6",
            76: "rib_right_7",
            77: "rib_right_8",
            78: "rib_right_9",
            79: "rib_right_10",
            80: "rib_right_11",
            81: "rib_right_12",
            82: "humerus_left",
            83: "humerus_right",
            84: "scapula_left",
            85: "scapula_right",
            86: "clavicula_left",
            87: "clavicula_right",
            88: "femur_left",
            89: "femur_right",
            90: "hip_left",
            91: "hip_right",
            92: "sacrum",
            93: "face",
            94: "gluteus_maximus_left",
            95: "gluteus_maximus_right",
            96: "gluteus_medius_left",
            97: "gluteus_medius_right",
            98: "gluteus_minimus_left",
            99: "gluteus_minimus_right",
            100: "autochthon_left",
            101: "autochthon_right",
            102: "iliopsoas_left",
            103: "iliopsoas_right",
            104: "urinary_bladder"
        }

        merged_organ_id = {
            'adrenal_gland_left': 0,
            'adrenal_gland_right': 0,
            'aorta': 1,
            'autochthon_left': 2,
            'autochthon_right': 2,
            'brain': 3,
            'clavicula_left': 4,
            'clavicula_right': 4,
            'colon': 5,
            'duodenum': 6,
            'esophagus': 7,
            'face': 8,
            'femur_left': 9,
            'femur_right': 9,
            'gallbladder': 10,
            'gluteus_maximus_left': 11,
            'gluteus_maximus_right': 11,
            'gluteus_medius_left': 11,
            'gluteus_medius_right': 11,
            'gluteus_minimus_left': 11,
            'gluteus_minimus_right': 11,
            'heart_atrium_left': 12,
            'heart_atrium_right': 12,
            'heart_myocardium': 12,
            'heart_ventricle_left': 12,
            'heart_ventricle_right': 12,
            'hip_left': 13,
            'hip_right': 13,
            'humerus_left': 14,
            'humerus_right': 14,
            'iliac_artery_left': 15,
            'iliac_artery_right': 15,
            'iliac_vena_left': 16,
            'iliac_vena_right': 16,
            'iliopsoas_left': 17,
            'iliopsoas_right': 17,
            'inferior_vena_cava': 18,
            'kidney_left': 19,
            'kidney_right': 19,
            'liver': 20,
            'lung_lower_lobe_left': 21,
            'lung_lower_lobe_right': 21,
            'lung_middle_lobe_right': 21,
            'lung_upper_lobe_left': 21,
            'lung_upper_lobe_right': 21,
            'pancreas': 22,
            'portal_vein_and_splenic_vein': 23,
            'pulmonary_artery': 24,
            'rib_left_1': 25,
            'rib_left_10': 25,
            'rib_left_11': 25,
            'rib_left_12': 25,
            'rib_left_2': 25,
            'rib_left_3': 25,
            'rib_left_4': 25,
            'rib_left_5': 25,
            'rib_left_6': 25,
            'rib_left_7': 25,
            'rib_left_8': 25,
            'rib_left_9': 25,
            'rib_right_1': 25,
            'rib_right_10': 25,
            'rib_right_11': 25,
            'rib_right_12': 25,
            'rib_right_2': 25,
            'rib_right_3': 25,
            'rib_right_4': 25,
            'rib_right_5': 25,
            'rib_right_6': 25,
            'rib_right_7': 25,
            'rib_right_8': 25,
            'rib_right_9': 25,
            'sacrum': 26,
            'scapula_left': 27,
            'scapula_right': 27,
            'small_bowel': 28,
            'spleen': 29,
            'stomach': 30,
            'trachea': 31,
            'urinary_bladder': 32,
            'vertebrae_C1': 33,
            'vertebrae_C2': 33,
            'vertebrae_C3': 33,
            'vertebrae_C4': 33,
            'vertebrae_C5': 33,
            'vertebrae_C6': 33,
            'vertebrae_C7': 33,
            'vertebrae_L1': 34,
            'vertebrae_L2': 34,
            'vertebrae_L3': 34,
            'vertebrae_L4': 34,
            'vertebrae_L5': 34,
            'vertebrae_T1': 35,
            'vertebrae_T2': 35,
            'vertebrae_T3': 35,
            'vertebrae_T4': 35,
            'vertebrae_T5': 35,
            'vertebrae_T6': 35,
            'vertebrae_T7': 35,
            'vertebrae_T8': 35,
            'vertebrae_T9': 35,
            'vertebrae_T10': 35,
            'vertebrae_T11': 35,
            'vertebrae_T12': 35
        }

        mask_ct = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask_ct)

        fused_mask = np.zeros_like(mask)
        for original_id, organ_name in class_map.items():
            if organ_name not in merged_organ_id:
                continue
            merged_id = merged_organ_id[organ_name]
            fused_mask[mask == original_id] = merged_id + 1

        fused_mask_sitk = sitk.GetImageFromArray(fused_mask)
        fused_mask_sitk.CopyInformation(mask_ct)
        
        save_merged_mask_path = os.path.join(root_dir, f"merged_masks/{relative_path}")
        sitk.WriteImage(fused_mask_sitk, save_merged_mask_path)

        # Step 2: Resize image and mask
        Path(os.path.join(root_dir, "resized_images/" + relative_path)).parent.mkdir(parents=True, exist_ok=True)
        Path(os.path.join(root_dir, "resized_masks/" + relative_path)).parent.mkdir(parents=True, exist_ok=True)

        image_path = mask_path.replace("/merlin_mask/", "/merlin_data/")
        mask_path = save_merged_mask_path

        data = {"image": image_path, "label": mask_path}
        res = transforms.LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True)(data)
        image = res["image"]

        affine = res["image_meta_dict"]["affine"]
        spacing = (abs(affine[0, 0].item()), abs(affine[1, 1].item()), abs(affine[2, 2].item()))
        _, h, w, d = image.shape

        ref_spacing = (1.0, 1.0, 5.0)
        scale = [spacing[i] / ref_spacing[i] for i in range(3)]
        target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]

        trans = transforms.Compose(
            [
                transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear"),
                transforms.Resized(spatial_size=target_size, keys=["label"], mode="nearest"),
                transforms.SaveImaged(
                    output_dir=Path(os.path.join(root_dir, "resized_images/" + relative_path)).parent,
                    keys=["image"],
                    output_postfix="",
                    separate_folder=False,
                    resample=False,
                ),
                transforms.SaveImaged(
                    output_dir=Path(os.path.join(root_dir, "resized_masks/" + relative_path)).parent,
                    keys=["label"],
                    output_postfix="",
                    separate_folder=False,
                    resample=False,
                ),
            ]
        )

        trans(res)

    except Exception as e:
        print(mask_path, e)


if "__main__" == __name__:
    import os
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from multiprocessing import Pool
    import json
    import numpy as np
    import random
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--slice-id", default=0, required=False, type=int)
    parser.add_argument("--num-slices", default=4, required=False, type=int)

    args = parser.parse_args()

    slice_id = args.slice_id
    num_slices = args.num_slices

    # get all img paths
    src_dir = '/Merlin/download/merlinabdominalctdataset/merlin_mask'  # get mask by TotalSegmentator
    mask_paths = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]

    img_paths = mask_paths[slice_id::num_slices]
    random.shuffle(img_paths)
    print(f'Num_slice: {num_slices}, Slice_id: {slice_id}, slice_num: {len(img_paths)}, Total_num: {len(mask_paths)}')


    with ProcessPoolExecutor(max_workers=36) as executor:
        for _ in tqdm(executor.map(fuse_mask, img_paths), total=len(img_paths)):
            pass
    
    print(f'----> done: slice_id: {slice_id}')


# sudo python process_img_mask.py --slice-id 0

