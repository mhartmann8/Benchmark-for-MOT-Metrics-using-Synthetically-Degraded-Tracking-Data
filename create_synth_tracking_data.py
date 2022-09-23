"""Utitlites to reproduce synthetically degraded segmentation masks
and lineage files with fragmentation, ID switch or mitosis tracking
errors."""
import datetime
from typing import Sequence
import os
from pathlib import Path
import time

import pandas as pd
import numpy as np

from pycocotools import mask as pycocomask
from tifffile.tifffile import imsave

from extract_data import export_masks, extract_segmentation_data, get_masks
from tracking_metrics import timing


def read_lineage_file(path: Path) -> pd.DataFrame:
    """Loads in a lineage file in the CTC format and returns it as a pd.DataFrame"""
    track_df = pd.read_csv(path, sep=" ", header=None)
    track_df = track_df.rename(columns={0: "id", 1: "start", 2: "end", 3: "parent"})
    track_df.index = track_df.id
    # Set the trackId once to the original maskId, this is used to preserve the trackID during fragmentation
    track_df["trackId"] = track_df.id
    return track_df


def select_mitosis_ids(lineage_df: pd.DataFrame, n_percent: int) -> np.array:
    """
    Selects n% of the mitosis events from a lineage file
    Args:
        lineage_df: pd.DataFrame, lineage file in the CTC format
        n_percent: int, percentage of mitosis events in lineage_df that will be selected

    Returns: np.array, selected parent IDs

    """
    # Extract all mask ids that are parents
    parent_masks = lineage_df.parent.unique()
    parent_masks = parent_masks[parent_masks != 0]

    # choose which mitosis events to modify based on percentage
    select_n_masks = int(np.rint(n_percent / 100 * len(parent_masks)))
    selected_parent_ids = np.random.choice(parent_masks, select_n_masks, replace=False)

    return selected_parent_ids


def missing_daughters_frames(segm_masks, lineage_df, n_percent, keep_parent_id=True):
    """
    Selects n% of the mitosis events from the lineage file and deletes the first frame of both daughters
    Args:
        segm_masks: dict containing all segmentation masks of an image sequence
        lineage_df: pd.DataFrame, lineage file in the CTC format
        n_percent: int, percentage of mitosis events in lineage_df that will be selected
        keep_parent_id: boolean, if true the parent column in the lineage_df is not modified, else it is set to 0

    Returns: segmentation masks, lineage DataFrame

    """
    selected_parent_ids = select_mitosis_ids(lineage_df, n_percent)
    for m_id in selected_parent_ids:
        daughters = lineage_df.loc[lineage_df.parent == m_id, "id"]

        for daughter_id in daughters:
            start_frame = lineage_df.loc[daughter_id, "start"]
            lineage_df.loc[daughter_id, "start"] = start_frame + 1

            segm_masks[start_frame].pop(daughter_id)

            if not keep_parent_id:
                lineage_df.loc[daughter_id, "parent"] = 0

    return segm_masks, lineage_df


def missing_daughter_single_frame(
    segm_masks, lineage_df, n_percent, keep_parent_id=True
):
    """
    Selects n% of the mitosis events from the lineage file and deletes the first frame of a single randomly chosen daughter
    Args:
        segm_masks: dict containing all segmentation masks of an image sequence
        lineage_df: pd.DataFrame, lineage file in the CTC format
        n_percent: int, percentage of mitosis events in lineage_df that will be selected
        keep_parent_id: boolean, if true the parent column in the lineage_df is not modified, else it is set to 0

    Returns: segmentation masks, lineage DataFrame

    """
    selected_parent_ids = select_mitosis_ids(lineage_df, n_percent)
    for m_id in selected_parent_ids:
        daughters = lineage_df.loc[lineage_df.parent == m_id, "id"]
        daughter_id = int(daughters.sample(1))

        start_frame = lineage_df.loc[daughter_id, "start"]
        lineage_df.loc[daughter_id, "start"] = start_frame + 1

        segm_masks[start_frame].pop(daughter_id)

        if not keep_parent_id:
            for d_id in daughters:
                lineage_df.loc[d_id, "parent"] = 0

    return segm_masks, lineage_df


def missing_parent_frame(segm_masks, lineage_df, n_percent, keep_parent_id=True):
    """
    Selects n% of the mitosis events from the lineage file and deletes the last frame of the parents
    Args:
        segm_masks: dict containing all segmentation masks of an image sequence
        lineage_df: pd.DataFrame, lineage file in the CTC format
        n_percent: int, percentage of mitosis events in lineage_df that will be selected
        keep_parent_id: boolean, if true the parent column in the lineage_df is not modified, else it is set to 0

    Returns: segmentation masks, lineage DataFrame

    """
    selected_parent_ids = select_mitosis_ids(lineage_df, n_percent)
    for m_id in selected_parent_ids:
        # Deletes the last frame of the randomly chosen parent cell
        end_frame = lineage_df.loc[m_id, "end"]
        lineage_df.loc[m_id, "end"] = end_frame - 1

        segm_masks[end_frame].pop(m_id)

        if not keep_parent_id:
            daughters = lineage_df.loc[lineage_df.parent == m_id, "id"]
            for daughter_id in daughters:
                lineage_df.loc[daughter_id, "parent"] = 0

    return segm_masks, lineage_df


def missing_single_mitosis_link(lineage_df: pd.DataFrame, n_percent) -> pd.DataFrame:
    """
    Selects n% of the mitosis events from the lineage file and deletes a single lineage link between one daughter and the parent.
    The segmentation masks are not passed, as only mitosis relations are modified.
    Args:
        lineage_df: pd.DataFrame, lineage file in the CTC format
        n_percent: int, percentage of mitosis events in lineage_df that will be selected

    Returns: lineage DataFrame

    """
    selected_parent_ids = select_mitosis_ids(lineage_df, n_percent)
    for m_id in selected_parent_ids:
        daughters = lineage_df.loc[lineage_df.parent == m_id, "id"]
        daughter_id = int(daughters.sample(1))

        lineage_df.loc[daughter_id, "parent"] = 0
        # segm_masks is unchanged by this, only mitosis relations are changed

    return lineage_df


def no_mitosis_links(lineage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all mitosis information from the lineage dataframe by setting the parent id to 0.
    Args:
        lineage_df: pd.DataFrame, lineage file in the CTC format

    Returns: lineage DataFrame

    """
    lineage_df["parent"] = 0
    return lineage_df


def id_switch_masks(segm_masks, lineage_df, select_n_percent):
    """
    Computes the centroids of masks and selects one of the 100 closest pairs. Then the compute_switch_masks is called to finalize the ID switch.
    This is repeated until the required percentage is reached.
    Args:
        segm_masks: dict containing all segmentation masks of an image sequence
        lineage_df: pd.DataFrame, lineage file in the CTC format
        select_n_percent: float indicating the fraction of segmentation masks to select

    Returns: list of selected mask pairs

    """
    # define pairs of segmentation masks that can be switched based
    # on their distance to each other
    initial_masks = get_masks(segm_masks)
    possible_select_pairs = {}  # (time, m_id): {(time, m_id) : cost ,....
    selected_masks_list = []
    all_centroids = {
        (t, m_id): np.mean(segm_masks[t][m_id], axis=-1) for t, m_id in initial_masks
    }
    for t, m_id in initial_masks:
        mask_centroid = all_centroids[(t, m_id)]
        possible_select_pairs[(t, m_id)] = {
            (t, other_id): np.linalg.norm(mask_centroid - all_centroids[(t, other_id)])
            for other_id, other_mask in segm_masks[t].items()
            if other_id != m_id
        }

    # Calculate the amount of Pairs that should be switched, based on the total amount of possible pairs
    select_n_pairs = np.rint((select_n_percent / 100) * (len(initial_masks) / 2))

    while (len(selected_masks_list) < select_n_pairs) and possible_select_pairs:
        # sample per track its best switch partner - based on the distance
        potential_merge = {}
        for mask_id, neighbors in possible_select_pairs.items():
            if not neighbors:
                continue
            other_masks, distances = list(zip(*[(k, v) for k, v in neighbors.items()]))
            min_distance = min(distances)
            closest_mask = other_masks[distances.index(min_distance)]

            if (
                mask_id[1] in segm_masks[mask_id[0]]
                and closest_mask[1] in segm_masks[closest_mask[0]]
            ):
                if (
                    lineage_df.loc[mask_id[1], "start"] <= closest_mask[0]
                    and lineage_df.loc[mask_id[1], "end"] >= closest_mask[0]
                ):
                    potential_merge[((mask_id, closest_mask))] = min_distance + 1

        # select closest 100 pairs and select one of these randomly
        pairs = [(k, v) for k, v in potential_merge.items()]
        pairs.sort(key=lambda x: x[1])
        distance_centroids = np.array([pair[1] for pair in pairs])
        index = np.argmax(distance_centroids > np.median(distance_centroids))
        pairs = pairs[: max(index, 100)]

        switch_pairs, dist = list(zip(*pairs))
        sampling_weight = 1 / np.array(dist)
        sampling_weight = np.array(sampling_weight) / np.sum(sampling_weight)

        selected_index = np.random.choice(len(switch_pairs), size=1, p=sampling_weight)[
            0
        ]
        selected_masks = switch_pairs[selected_index]

        segm_masks, lineage_df = compute_switch_masks(
            segm_masks, lineage_df, selected_masks
        )

        selected_masks_list.append(selected_masks)
        possible_select_pairs.pop(selected_masks[0])

    print(len(selected_masks_list), select_n_pairs, len(initial_masks))

    print(
        f"Percent Mask Id Switches: {100*len(selected_masks_list)/(len(initial_masks)/2):.2f} %"
    )

    return segm_masks, lineage_df, selected_masks_list


def compute_switch_masks(segm_masks, lineage_df, selected_masks):
    """
    Computes a single id switch at a selected and all following timepoints of two mask IDs.
    At each time step the Mask IDs are either switched or renamed if only one mask is visible.
    Any conflicts with mitosis events is checked and both the segmentation masks and the lineage_df are updated.
    Args:
        segm_masks: dict containing all segmentation masks of an image sequence
        lineage_df: pd.DataFrame, lineage file in the CTC format
        selected_masks: ((t,m1), (t, m2)), tuple for the selected pair with each holding the timepoint of the switch timepoint and the mask id

    Returns: segmentation masks, lineage DataFrame

    """
    initial_keys = {k: list(v.keys()) for k, v in segm_masks.items()}
    switch_ids = (selected_masks[0][1], selected_masks[1][1])
    timepoints_after_switch = list(initial_keys.keys())[selected_masks[0][0] :]

    for t in timepoints_after_switch:
        m0_isvisible = switch_ids[0] in initial_keys[t]
        m1_isvisible = switch_ids[1] in initial_keys[t]
        count_visible = m0_isvisible + m1_isvisible

        if count_visible == 2:
            ## Switch masks between mask_0 and mask_1
            mask_0 = segm_masks[t].pop(switch_ids[0])
            mask_1 = segm_masks[t].pop(switch_ids[1])

            segm_masks[t][switch_ids[0]] = mask_1
            segm_masks[t][switch_ids[1]] = mask_0

        elif count_visible == 1:
            if m0_isvisible:
                ## Rename mask_0 to mask_1
                mask_0 = segm_masks[t].pop(switch_ids[0])
                segm_masks[t][switch_ids[1]] = mask_0

            elif m1_isvisible:
                # Rename mask_1 to mask_0
                mask_1 = segm_masks[t].pop(switch_ids[1])
                segm_masks[t][switch_ids[0]] = mask_1

    # Switching the end frames in the lineage file
    end_m0 = lineage_df.loc[switch_ids[0], "end"]
    end_m1 = lineage_df.loc[switch_ids[1], "end"]
    lineage_df.loc[switch_ids[0], "end"] = end_m1
    lineage_df.loc[switch_ids[1], "end"] = end_m0

    # Reassigning the children to the switched track
    m0_descendants = lineage_df.loc[lineage_df.parent == switch_ids[0], "parent"]
    m1_descendants = lineage_df.loc[lineage_df.parent == switch_ids[1], "parent"]
    if not m0_descendants.empty:
        lineage_df.loc[m0_descendants.index, "parent"] = switch_ids[1]
    if not m1_descendants.empty:
        lineage_df.loc[m1_descendants.index, "parent"] = switch_ids[0]

    return segm_masks, lineage_df


def get_trajectories(segm_masks):
    """
    Computes the complete trajectories for the passed segmentation masks.
    Args:
        segm_masks: dict containing all segmentation masks of an image sequence

    Returns: dict{m_id : [(t1, m_id), (t2, m_id), ...]}, trajectories

    """
    trajectories = dict()
    for t, masks_at_t in segm_masks.items():
        for m_id in masks_at_t.keys():
            trajectories.setdefault(m_id, [])
            trajectories[m_id].append(tuple((t, m_id)))
    return trajectories


def get_connected_tracklets(traj):
    """
    Split a trajectory with gaps into tracklets of consecutive timepoints.
    Args:
        trajectory: np.array, sequence of timepoints

    Returns: array of arrays with consecutive timepoints

    """
    """connects consecutive timesteps"""
    traj_times = [t for t, _ in traj]
    return np.split(traj_times, np.where(np.diff(traj_times) != 1)[0] + 1)


def fragmentation(
    segm_masks: dict,
    lineage_df: pd.DataFrame,
    frag_n_percent: int,
    mean_frag_len: int = None,
    set_frag_parent: bool = True,
):
    """
    Create fragmented tracks by sampling from a 2 state Markov model the points to delete. The Markov model is initialised according
    to the required fragmentation percentage and gap length. This allows customizability and variation in how the fragmentation is distributed.
    Args:
        segm_masks: dict containing all segmentation masks of an image sequence
        lineage_df: pd.DataFrame, lineage file in the CTC format
        frag_n_percent: int, percentage of fragmentation of the trajectories
        mean_frag_len: int, specify how long the continuous gaps introduced by the fragmentation should be
        set_frag_parent: boolean, if true the lineage_df is used to store the connection between tracklets of the same trajectory,
                         by setting the parent ID to the ID of the previous segment after a fragmentation, else it is set to 0

    Returns: new fragmented segmentation masks, updated lineage DataFrame, fragged_trajectories as dict

    """
    initial_trajectories = get_trajectories(segm_masks)
    p_frag = frag_n_percent / 100
    if p_frag < 0.01:
        p_frag = 0.01
    elif p_frag > 0.9:
        p_frag = 0.9

    # Calculate the transition probabilites of the 2 state Markov model
    # The probability b to switch back to the good quality tracking state is connected to the desired fragment length.
    # This is used to calculate the probability a to switch back to the bad quality tracking state, as a and b are linked
    # through the equilibrium state of a Markov model.
    if mean_frag_len:
        b = 1 / mean_frag_len
        a = (p_frag * b) / (1 - p_frag)
    else:
        mean_frag_len = 1 / (1 - p_frag)
        b = 1 / mean_frag_len
        a = mean_frag_len * b - b

    print(
        f"\n\np_frag= {100*p_frag:2.1f}%, E[T_G] = {mean_frag_len:3.1f}, a= {100*a:2.1f} %,  b= {100*b:2.1f} %\n"
    )

    # Sampling which points of the trajectory are deleted and applying those changes to the segm_masks
    fragged_trajectories = dict()
    for m_id, traj in initial_trajectories.items():
        emission_per_state = markov_sample(len(traj), p_frag, a, b)
        frag_mask = np.where(emission_per_state)[0]
        frag_index = np.array(traj)[frag_mask]

        fragged_trajectories.setdefault(m_id, [])
        fragged_trajectories[m_id].extend(frag_index)

        for t, m in frag_index:
            segm_masks[t].pop(m)

    print(
        "\nPercent Fragmentation: ",
        f"{100*sum([len(points) for points in fragged_trajectories.values()]) / sum([len(traj) for traj in initial_trajectories.values()]):.2f} %",
        "\n",
    )

    trajectories = get_trajectories(segm_masks)
    new_lines, new_masks = [], dict()
    initial_ids, used_ids = set(trajectories.keys()), set()
    new_id = 1

    # Create a Dataframe for parent connections, that is updated with the most frequent fragments of the parent if set_frag_parent is True
    parent_df = lineage_df.drop(columns=["start", "id", "end"])
    parent_df.index = parent_df.trackId
    parent_df = parent_df.drop(columns="trackId")

    for track_id, traj in sorted(trajectories.items()):
        new_id = track_id
        for tr in get_connected_tracklets(traj):
            traj_start, traj_end = tr[0], tr[-1]

            # Handle edge case where the complete parent track is missing, sets parent id to 0, only for first tracklet of a track
            parent_id = parent_df.loc[track_id, "parent"]
            if new_id == track_id and parent_id not in trajectories.keys():
                parent_id = 0

            # Update new Lineage File and Segm Masks
            newEntry = [new_id, traj_start, traj_end, parent_id, track_id]
            new_lines.append(newEntry)
            for t in range(traj_start, traj_end + 1):
                temp_mask = segm_masks[t].pop(track_id)
                new_masks.setdefault(t, {})
                new_masks[t][new_id] = temp_mask

            # In ctc format, fragmentation is handled by setting the parent id to the last tracklet id
            # Real parent-child relationships are kept by updating the most recent tracklet m_id of a gt_parent track as the parent of a child
            if set_frag_parent:
                parent_df.loc[track_id] = new_id
            else:
                # in the other case only the first fragment recieves the parentId and following are set to 0
                parent_df.loc[track_id] = 0

            used_ids.add(new_id)
            new_id = max(initial_ids | used_ids) + 1

        parent_df.loc[parent_df.parent == track_id, "parent"] = parent_df.loc[
            track_id, "parent"
        ]

    new_lines_df = pd.DataFrame(new_lines, columns=lineage_df.columns)
    new_lines_df = new_lines_df.sort_values(["trackId", "start"])
    new_lines_df.index = new_lines_df.id

    return new_masks, new_lines_df, fragged_trajectories


def markov_sample(sample_length: int, p_frag, a, b):
    """
    Represents a 2 state Markov model initialised with a transition matrix made up of a and b,
    the transition probabilities from state 1 to state 2 and back. A sequence is sampled,
    where a targeted fragmentation percentage is defined as the amount of switches from the
    good quality tracking state 1 to the bad tracking quality state 2, which leads to the deletion
    of the trajectory points that overlap with a 1 in the emission sequence list.
    Args:
        sample_length: int, the amount of markov states to sample
        p_frag: float, percentage of fragmentation between 0 and 1
        a: float, the probability of switching to the bad tracking state 2
        b: float, the probability of switching to the good tracking state 1

    Returns: np.array, emissions of the markov model with a 1 if the point should be deleted

    """
    p_emission = [0.001, 0.999]

    # a is p01, b is p10
    A = np.array([[1 - a, a], [b, 1 - b]])

    # Init the start states to the desired equilibrium state
    curr_state = np.random.choice([0, 1], p=[1 - p_frag, p_frag])
    state_seq = [curr_state]
    emission_seq = [
        np.random.choice([0, 1], p=[1 - p_emission[curr_state], p_emission[curr_state]])
    ]

    # Sample the sequence using transition probabilities of the current state (0 is keep, 1 is frag)
    for _ in range(sample_length - 1):
        curr_state = np.random.choice([0, 1], p=A[curr_state])
        state_seq.append(curr_state)
        emission_seq.append(
            np.random.choice(
                [0, 1], p=[1 - p_emission[curr_state], p_emission[curr_state]]
            )
        )

    return np.array(emission_seq)


def mask_coords_to_img(mask_coords, img_shape: Sequence[int]) -> np.array:
    """Creates a numpy array in the specified shape with mask indexes at the passed coordinates."""
    img = np.zeros(img_shape, dtype=np.uint16)
    mask_indices = np.array(mask_coords)
    valid_index = mask_indices < np.array(img_shape).reshape(-1, 1)
    mask_indices = tuple(mask_indices[:, np.all(valid_index, axis=0)])
    img[mask_indices] = 1
    return img.astype(np.uint8)


def compress_mask_pycoco(segm_mask, img_shape: Sequence[int]):
    """
    Uses the image processing library pycocotools to create a compressed string version
    of the segmentation masks using the Run Length Encoding (RLE) format.
    This is necessery for the Trackeval RobMots format.
    (https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)
    """
    processed_mask = np.asfortranarray(mask_coords_to_img(segm_mask, img_shape))
    compressed_rle = pycocomask.encode(processed_mask)
    return compressed_rle


def export_ctc(segm_masks, lineage_df, export_path, img_size):
    """Saves the data set in the file format of the Cell Tracking Challenge."""
    # save the images with export masks and the tracking file as txt
    os.makedirs(export_path, exist_ok=True)

    # If all masks in an image are fragmented, add an empty image to the folder so aogm can be calculated
    t_max = max(3, np.ceil(np.log10(max(segm_masks.keys()))))
    for t in range(max(segm_masks.keys()) + 1):
        if t not in segm_masks:
            img = np.zeros(img_size, dtype=np.uint16)
            img_name = "mask" + str(t).zfill(t_max) + ".tif"
            imsave(export_path / img_name, img.astype(np.uint16), compress=2)

    # then export the lineage file and the masks
    export_masks(
        segm_masks, export_path, img_size
    )  # .sort_values(["trackId", "start"]) # .drop(columns="trackId")
    lineage_df.drop(columns="trackId").to_csv(
        export_path / "res_track.txt", sep=" ", index=False, header=False
    )


def export_robmots_data(
    segm_masks: dict, lineage_df: pd.DataFrame, img_shape, folder_path, gt=False
):
    """
    This creates the folder paths in the RobMots format needed to use the Trackeval evaluation package.
    The evaluation of MOTA, IDF1 and HOTA is done on these, as they do not support the standard file format
    used by the Cell Tracking Challenge.
    Args:
        segm_masks: dict containing all segmentation masks of an image sequence
        lineage_df: pd.DataFrame, lineage file in the CTC format
        img_shape:
        folder_path:
        gt:
    """

    GT_PATH = r"gt\train\bdd_mots\data"
    RES_PATH = r"trackers\train\M\data\bdd_mots"
    SEQNAME = "0000"
    save_path = folder_path / GT_PATH if gt else folder_path / RES_PATH
    os.makedirs(save_path, exist_ok=True)

    if gt:
        with open(save_path.parent / "seqmap.txt", "w") as f:
            f.write(f"{SEQNAME} {len(segm_masks.keys())} {img_shape[0]} {img_shape[1]}")

        with open(save_path.parent / "clsmap.txt", "w") as f:
            f.write(str(1))

    robMots_save_file = []
    for _, track in lineage_df.iterrows():
        for t in range(track.start, track.end + 1):
            # print(t, track, track.id in segm_masks[t])
            mask_coords = segm_masks[t][track.id]
            compressed_rle = compress_mask_pycoco(mask_coords, img_shape)
            newEntry = f'{t} {track.trackId} 1 1 {compressed_rle["size"][0]} {compressed_rle["size"][1]} {compressed_rle["counts"].decode()}\n'
            robMots_save_file.append(newEntry)

    with open(save_path / (SEQNAME + ".txt"), "w") as f:
        for i, line in enumerate(robMots_save_file):
            if i + 1 == len(robMots_save_file):
                f.write(line[:-1])
            else:
                f.write(line)


def combine_track_errors(segm_masks, lineage_df, n_percent_errors, set_parent):
    """
    Creates a data set with a combination of three tracking errors: n%/3 of ID switch, n%/3 of fragmentation
    and n%/3 of missing daughter frames errors.
    Args:
        segm_masks: dict containing all segmentation masks of an image sequence
        lineage_df: pd.DataFrame, lineage file in the CTC format
        n_percent_errors: int, total percentage of errors, is split on 3 error types
        set_parent: boolean, used to toggle retaining information of predecessor for fragmentation and mitosis errors

    Returns: new fragmented segmentation masks, updated lineage DataFrame

    """
    final_masks = {}
    n_masks_initial = len(get_masks(segm_masks))
    # split error percentage evenly over split, merge, remove errors
    for i in range(3):

        n_masks_remaining = len(get_masks(segm_masks))
        n_percent_faction = n_masks_initial / n_masks_remaining * n_percent_errors / 3
        if i == 0:
            segm_masks, lineage_df, selected_masks = id_switch_masks(
                segm_masks, lineage_df, n_percent_faction
            )
        elif i == 1:
            segm_masks, lineage_df = missing_daughters_frames(
                segm_masks, lineage_df, n_percent_faction, keep_parent_id=True
            )

        else:
            segm_masks, lineage_df, selected_masks = fragmentation(
                segm_masks, lineage_df, n_percent_faction, set_frag_parent=True
            )

    return segm_masks, lineage_df


@timing
def set_up_synth_track_errors(data_path, results_path, percentages, n_runs):
    """
    Creates synthetically degraded tracking results for a large selection of different tracking error scenarios.
    A seed can be specified so each tracking error can be run individually.
    Args:
        data_path: path to ground truth segmentation images
        results_path: path where to save segmentation images with added errors
        percentages: list containing percentages of segmentation errors
        n_runs: int indicating the number of experiments to run per error type
                and tracking error percentage
    """
    all_perc_passes = 2 * len(percentages)
    all_start_time = time.monotonic()

    for i in range(1, 3):
        d_path = data_path / (str(i).zfill(2) + "_GT") / "TRA"
        if not d_path.is_dir():
            continue
        print(data_path.name, str(i).zfill(2))

        for j, n_percent in enumerate(percentages):
            if j > 0:
                time_left = str(
                    datetime.timedelta(
                        seconds=((all_perc_passes - j - 1) * single_perc_runtime)
                    )
                )[:-7]
                print(
                    f"Editing {n_percent}%, Edit Time so far: {str(datetime.timedelta(seconds=(time.monotonic() - all_start_time)))[:-7]}, Estimated Time left: {time_left}"
                )
            else:
                print(f"Editing {n_percent}%")

            start_time = time.monotonic()
            for n in range(n_runs):
                print(f"run {n+1}/{n_runs}")
                # Here the tracking error methods are called, and results saved
                lineage_df = read_lineage_file(d_path / "man_track.txt")
                segm_masks, img_size = extract_segmentation_data(d_path)
                lineage_df = no_mitosis_links(lineage_df)
                export_path = (
                    results_path
                    / (str(i).zfill(2))
                    / "noMitosis"
                    / ("percentage_" + str(n_percent))
                    / ("run_" + str(n))
                )
                export_ctc(segm_masks, lineage_df, export_path, img_size)
                export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                lineage_df = read_lineage_file(d_path / "man_track.txt")
                segm_masks, img_size = extract_segmentation_data(d_path)
                lineage_df = missing_single_mitosis_link(lineage_df, n_percent)
                export_path = (
                    results_path
                    / (str(i).zfill(2))
                    / "missing_single_mitosis_link"
                    / ("percentage_" + str(n_percent))
                    / ("run_" + str(n))
                )
                export_ctc(segm_masks, lineage_df, export_path, img_size)
                export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                lineage_df = read_lineage_file(d_path / "man_track.txt")
                segm_masks, img_size = extract_segmentation_data(d_path)
                segm_masks, lineage_df = missing_daughter_single_frame(
                    segm_masks, lineage_df, n_percent, keep_parent_id=True
                )
                export_path = (
                    results_path
                    / (str(i).zfill(2))
                    / "missing_single_daughter_frame_keepInheritence"
                    / ("percentage_" + str(n_percent))
                    / ("run_" + str(n))
                )
                export_ctc(segm_masks, lineage_df, export_path, img_size)
                export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                lineage_df = read_lineage_file(d_path / "man_track.txt")
                segm_masks, img_size = extract_segmentation_data(d_path)
                segm_masks, lineage_df = missing_daughter_single_frame(
                    segm_masks, lineage_df, n_percent, keep_parent_id=False
                )
                export_path = (
                    results_path
                    / (str(i).zfill(2))
                    / "missing_single_daughter_frame_noInheritence"
                    / ("percentage_" + str(n_percent))
                    / ("run_" + str(n))
                )
                export_ctc(segm_masks, lineage_df, export_path, img_size)
                export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                lineage_df = read_lineage_file(d_path / "man_track.txt")
                segm_masks, img_size = extract_segmentation_data(d_path)
                segm_masks, lineage_df = missing_parent_frame(
                    segm_masks, lineage_df, n_percent, keep_parent_id=True
                )
                export_path = (
                    results_path
                    / (str(i).zfill(2))
                    / "missing_last_parent_frame_keepInheritence"
                    / ("percentage_" + str(n_percent))
                    / ("run_" + str(n))
                )
                export_ctc(segm_masks, lineage_df, export_path, img_size)
                export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                lineage_df = read_lineage_file(d_path / "man_track.txt")
                segm_masks, img_size = extract_segmentation_data(d_path)
                segm_masks, lineage_df = missing_parent_frame(
                    segm_masks, lineage_df, n_percent, keep_parent_id=False
                )
                export_path = (
                    results_path
                    / (str(i).zfill(2))
                    / "missing_last_parent_frame_noInheritence"
                    / ("percentage_" + str(n_percent))
                    / ("run_" + str(n))
                )
                export_ctc(segm_masks, lineage_df, export_path, img_size)
                export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                lineage_df = read_lineage_file(d_path / "man_track.txt")
                segm_masks, img_size = extract_segmentation_data(d_path)
                segm_masks, lineage_df = missing_daughters_frames(
                    segm_masks, lineage_df, n_percent, keep_parent_id=True
                )
                export_path = (
                    results_path
                    / (str(i).zfill(2))
                    / "missing_daughters_frames_keepInheritence"
                    / ("percentage_" + str(n_percent))
                    / ("run_" + str(n))
                )
                export_ctc(segm_masks, lineage_df, export_path, img_size)
                export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                lineage_df = read_lineage_file(d_path / "man_track.txt")
                segm_masks, img_size = extract_segmentation_data(d_path)
                segm_masks, lineage_df = missing_daughters_frames(
                    segm_masks, lineage_df, n_percent, keep_parent_id=False
                )
                export_path = (
                    results_path
                    / (str(i).zfill(2))
                    / "missing_daughters_frames_noInheritence"
                    / ("percentage_" + str(n_percent))
                    / ("run_" + str(n))
                )
                export_ctc(segm_masks, lineage_df, export_path, img_size)
                export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                lineage_df = read_lineage_file(d_path / "man_track.txt")
                segm_masks, img_size = extract_segmentation_data(d_path)
                segm_masks, lineage_df, selected_masks = id_switch_masks(
                    segm_masks, lineage_df, n_percent
                )
                export_path = (
                    results_path
                    / (str(i).zfill(2))
                    / "id_switch"
                    / ("percentage_" + str(n_percent))
                    / ("run_" + str(n))
                )
                export_ctc(segm_masks, lineage_df, export_path, img_size)
                export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                lineage_df = read_lineage_file(d_path / "man_track.txt")
                segm_masks, img_size = extract_segmentation_data(d_path)
                segm_masks, lineage_df, fragged_trajs = fragmentation(
                    segm_masks,
                    lineage_df,
                    n_percent,
                    mean_frag_len=None,
                    set_frag_parent=True,
                )
                export_path = (
                    results_path
                    / (str(i).zfill(2))
                    / "fragmentation_noFragLen_setParent"
                    / ("percentage_" + str(n_percent))
                    / ("run_" + str(n))
                )
                export_ctc(segm_masks, lineage_df, export_path, img_size)
                export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                lineage_df = read_lineage_file(d_path / "man_track.txt")
                segm_masks, img_size = extract_segmentation_data(d_path)
                segm_masks, lineage_df, fragged_trajs = fragmentation(
                    segm_masks,
                    lineage_df,
                    n_percent,
                    mean_frag_len=None,
                    set_frag_parent=False,
                )
                export_path = (
                    results_path
                    / (str(i).zfill(2))
                    / "fragmentation_noFragLen_noParent"
                    / ("percentage_" + str(n_percent))
                    / ("run_" + str(n))
                )
                export_ctc(segm_masks, lineage_df, export_path, img_size)
                export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                # compute the fragmentation for a small (5), medium (25), and large (50) fragmentation gap length
                for frag_len in [5, 25, 50]:
                    lineage_df = read_lineage_file(d_path / "man_track.txt")
                    segm_masks, img_size = extract_segmentation_data(d_path)
                    segm_masks, lineage_df, fragged_trajs = fragmentation(
                        segm_masks,
                        lineage_df,
                        n_percent,
                        mean_frag_len=frag_len,
                        set_frag_parent=True,
                    )
                    export_path = (
                        results_path
                        / (str(i).zfill(2))
                        / f"fragmentation_{frag_len}FragLen_setParent"
                        / ("percentage_" + str(n_percent))
                        / ("run_" + str(n))  # + f"_{n_percent}FragLen"
                    )
                    export_ctc(segm_masks, lineage_df, export_path, img_size)
                    export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                    lineage_df = read_lineage_file(d_path / "man_track.txt")
                    segm_masks, img_size = extract_segmentation_data(d_path)
                    segm_masks, lineage_df, fragged_trajs = fragmentation(
                        segm_masks,
                        lineage_df,
                        n_percent,
                        mean_frag_len=frag_len,
                        set_frag_parent=False,
                    )
                    export_path = (
                        results_path
                        / (str(i).zfill(2))
                        / f"fragmentation_{frag_len}FragLen_noParent"
                        / ("percentage_" + str(n_percent))
                        / ("run_" + str(n))  # + f"_{n_percent}FragLen"
                    )
                    export_ctc(segm_masks, lineage_df, export_path, img_size)
                    export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                lineage_df = read_lineage_file(d_path / "man_track.txt")
                segm_masks, img_size = extract_segmentation_data(d_path)
                segm_masks, lineage_df = combine_track_errors(
                    segm_masks, lineage_df, n_percent, set_parent=True
                )
                export_path = (
                    results_path
                    / (str(i).zfill(2))
                    / f"mixed_track_errors_setParent"
                    / ("percentage_" + str(n_percent))
                    / ("run_" + str(n))  # + f"_{n_percent}FragLen"
                )
                export_ctc(segm_masks, lineage_df, export_path, img_size)
                export_robmots_data(segm_masks, lineage_df, img_size, export_path)

                lineage_df = read_lineage_file(d_path / "man_track.txt")
                segm_masks, img_size = extract_segmentation_data(d_path)
                segm_masks, lineage_df = combine_track_errors(
                    segm_masks, lineage_df, n_percent, set_parent=False
                )
                export_path = (
                    results_path
                    / (str(i).zfill(2))
                    / f"mixed_track_errors_noParent"
                    / ("percentage_" + str(n_percent))
                    / ("run_" + str(n))  # + f"_{n_percent}FragLen"
                )
                export_ctc(segm_masks, lineage_df, export_path, img_size)
                export_robmots_data(segm_masks, lineage_df, img_size, export_path)

            single_perc_runtime = time.monotonic() - start_time
