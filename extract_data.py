"""Utilities for file managment, extracting segmentation masks and computing centroids."""
import os
import re
import numpy as np
import pandas as pd
from scipy.ndimage.morphology import distance_transform_edt
from tifffile import imread, imsave


def get_indices_pandas(data, background_id=0):
    """
    Extracts for each mask id its positions within the array.
    Args:
        data: a np. array with masks, where all pixels belonging to the
            same masks have the same integer value
        background_id: integer value of the background

    Returns: data frame: indices are the mask id , values the positions of the mask pixels

    """
    if data.size < 1e9:  # aim for speed at cost of high memory consumption
        masked_data = data != background_id
        flat_data = data[masked_data]  # d: data , mask attribute
        dummy_index = np.where(masked_data.ravel())[0]
        df = pd.DataFrame.from_dict({"mask_id": flat_data, "flat_index": dummy_index})
        df = df.groupby("mask_id").apply(
            lambda x: np.unravel_index(x.flat_index, data.shape)
        )
    else:  # aim for lower memory consumption at cost of speed
        flat_data = data[(data != background_id)]  # d: data , mask attribute
        dummy_index = np.where((data != background_id).ravel())[0]
        mask_indices = np.unique(flat_data)
        df = {"mask_id": [], "index": []}
        data_shape = data.shape
        for mask_id in mask_indices:
            df["index"].append(
                np.unravel_index(dummy_index[flat_data == mask_id], data_shape)
            )
            df["mask_id"].append(mask_id)
        df = pd.DataFrame.from_dict(df)
        df = df.set_index("mask_id")
        df = df["index"].apply(lambda x: x)  # convert to same format as for other case
    return df


def get_img_files(img_dir, starts_with=""):
    """
    Extracts a set of tiff files from a folder.
    Args:
        img_dir: path to the image folder
        starts_with: optional string the image name needs to start with

    Returns:

    """
    img_file_pattern = re.compile(
        r"(\D*)(\d+)(\D*)\.(" + "|".join(("tif", "tiff")) + ")"
    )
    files = {
        int(img_file_pattern.match(file).groups()[1]): (img_dir / file).as_posix()
        for file in os.listdir(img_dir)
        if file.endswith(("tif", "tiff")) and file.startswith(starts_with)
    }
    return files


def extract_segmentation_data(data_path):
    """Extracts all segmentation masks from a sequence of images."""
    images = get_img_files(data_path)
    segmentation_masks = {
        t: get_indices_pandas(imread(img)) for t, img in images.items()
    }
    img_size = imread(images[list(images.keys())[0]]).shape
    return segmentation_masks, img_size


def get_masks(segm_masks):
    return [(time, m_id) for time, masks in segm_masks.items() for m_id in masks.keys()]


def compute_seeds(mask, n_seeds):
    """Computes seed points to split a segmentation mask."""
    mask = np.array(mask)

    box_shape = np.max(mask, axis=1) - np.min(mask, axis=1) + 3  # add background border
    dummy = np.zeros(tuple(box_shape))
    dummy[tuple(mask - np.min(mask, axis=1).reshape(-1, 1) + 1)] = 1
    dist = distance_transform_edt(dummy)
    stacked = np.stack(np.gradient(dist))
    abs_grad = np.sum(stacked**2, axis=0)
    seed_points = np.where((abs_grad < 0.1) & (dist > 0))
    if len(seed_points[0]) < n_seeds:
        seed_points = tuple(mask)
    else:
        # compute non shifted position
        seed_points = np.array(seed_points) + np.min(mask, axis=1).reshape(-1, 1) - 1
    seed_index = np.random.choice(len(seed_points[0]), n_seeds, replace=False)
    seed_points = np.array(seed_points)[..., seed_index]
    return seed_points


def export_masks(segm_masks, export_path, img_size):
    """Exports segmentation mask to sequences of segmentation images."""
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    t_max = max(3, np.ceil(np.log10(max(segm_masks.keys()))))
    for t, masks in segm_masks.items():
        img = np.zeros(img_size, dtype=np.uint16)
        for m_id, mask_indices in masks.items():
            mask_indices = np.array(mask_indices)
            valid_index = mask_indices < np.array(img_size).reshape(-1, 1)
            mask_indices = tuple(mask_indices[:, np.all(valid_index, axis=0)])
            img[mask_indices] = m_id

        img_name = "mask" + str(t).zfill(t_max) + ".tif"
        imsave(export_path / img_name, img.astype(np.uint16), compress=2)
