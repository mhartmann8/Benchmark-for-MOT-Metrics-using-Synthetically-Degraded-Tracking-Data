"""
Skript to simulate images with cells that correspond to a predefined lineage. Can be used to test methods on a 
reduced data set, where changes made by creat_synth_tracking_data.py methods can be verified manually. 
Due to the small data set size the methods may not work as intended when small percentages are selected. 
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import disk
from tifffile.tifffile import imsave

from utils import get_data_path


LINEAGE_GT = {
    "id": [1, 2, 3, 4, 5, 6, 7],
    "start": [0, 0, 9, 9, 11, 11, 12],
    "end": [8, 10, 15, 13, 19, 13, 20],
    "parent": [0, 0, 1, 1, 2, 2, 0],
}

IMG_SHAPE = (650, 625)
RADIUS = 25
EDGE_OFFSET = 2.5

# Create the sim data directories
PATH = Path(get_data_path() / "simulated-test-data")
d_path = PATH / ("01")
d_path_gt = PATH / ("01_GT") / "TRA"
d_path.mkdir(parents=True, exist_ok=True)
d_path_gt.mkdir(parents=True, exist_ok=True)

# add the man_track.txt file
with open(d_path_gt / "man_track.txt", "w") as file:
    for i, id in enumerate(LINEAGE_GT["id"]):
        file.write(
            f'{LINEAGE_GT["id"][i]} {LINEAGE_GT["start"][i]} {LINEAGE_GT["end"][i]} {LINEAGE_GT["parent"][i]}'
        )
        if id < max(LINEAGE_GT["id"]):
            file.write("\n")

# create segm_masks based on the lineage file
df = pd.DataFrame(LINEAGE_GT)
df.index = df.id
segm_masks = dict()
for t in range(max(LINEAGE_GT["end"]) + 1):
    segm_masks[t] = dict()
    original_masks = np.zeros(tuple(IMG_SHAPE))
    for m_id in df.id:
        if df.loc[m_id, "start"] <= t and df.loc[m_id, "end"] >= t:
            # cell is visible in this timestep
            offset = np.array(
                [
                    np.random.randint(
                        EDGE_OFFSET * RADIUS, IMG_SHAPE[0] - EDGE_OFFSET * RADIUS
                    ),
                    np.random.randint(
                        EDGE_OFFSET * RADIUS, IMG_SHAPE[1] - EDGE_OFFSET * RADIUS
                    ),
                ]
            )

            structure = disk(RADIUS)
            structure_coords = np.where(structure == 1)

            structure_coords = np.array(structure_coords)
            structure_coords[0] = structure_coords[0] + offset[0]
            structure_coords[1] = structure_coords[1] + offset[1]

            original_masks[tuple([structure_coords[0], structure_coords[1]])] = 1
            segm_masks[t][m_id] = tuple([structure_coords[0], structure_coords[1]])

# from ba_utils_temp import pretty_segm_masks
# pretty_segm_masks(segm_masks)
def modified_export_masks(segm_masks, export_path, img_size, file_name):
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

        img_name = file_name + str(t).zfill(t_max) + ".tif"
        imsave(export_path / img_name, img.astype(np.uint16), compress=2)


modified_export_masks(segm_masks, d_path, IMG_SHAPE, file_name="t")
modified_export_masks(
    segm_masks, d_path_gt.parent / "SEG", IMG_SHAPE, file_name="man_seg"
)
modified_export_masks(segm_masks, d_path_gt, IMG_SHAPE, file_name="man_track")
