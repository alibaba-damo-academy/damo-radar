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


def fuse_mask(mask_path, root_dir):
    # print('mask_path', mask_path)
    try:
        # Resize image and mask
        relative_path = "/".join(mask_path.split("/")[-1:])
        save_folder = "resized_images"
        
        Path(os.path.join(root_dir, save_folder, relative_path)).parent.mkdir(parents=True, exist_ok=True)

        image_path = mask_path.replace("/resized_masks/", "/merlin_data/")  # assumes merlin_data/ and resized_masks/ are located at the same directory level

        data = {"image": image_path}
        res = transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True)(data)
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
                transforms.SaveImaged(
                    output_dir=Path(os.path.join(root_dir, save_folder, relative_path)).parent,
                    keys=["image"],
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
    parser.add_argument("--num-slices", default=1, required=False, type=int)
    parser.add_argument("--src-dir", default=None, required=True, type=str, help='Path to our processed masks')
    parser.add_argument("--root-dir", default=None, required=True, type=str, help='Path to the directory containing the original Merlin data.')

    args = parser.parse_args()

    slice_id = args.slice_id
    num_slices = args.num_slices
    src_dir = args.src_dir
    root_dir = args.root_dir

    # get all img paths
    mask_paths = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]

    img_paths = mask_paths[slice_id::num_slices]
    random.shuffle(img_paths)
    print(f'Num_slice: {num_slices}, Slice_id: {slice_id}, slice_num: {len(img_paths)}, Total_num: {len(mask_paths)}')
    
    for p in img_paths:
        fuse_mask(p, root_dir)
    
    print(f'----> done: slice_id: {slice_id}')


# sudo python process_img.py

