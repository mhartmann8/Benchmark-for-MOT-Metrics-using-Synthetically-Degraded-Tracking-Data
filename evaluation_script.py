"""Script for a complete workflow of creating and evaluating synthetically degraded
tracking results."""
import os
import sys
from pathlib import Path
from typing import Iterable
import pandas as pd
import numpy as np

from utils import collect_leaf_paths, get_results_path, get_data_path
from extract_data import extract_segmentation_data
from evaluation_plots import create_all_plots, plot_trackeval_results
from tracking_metrics import (
    calc_ctc_scores,
    calculate_trackeval_scores,
    extract_aogm_info,
    timing,
)
from create_synth_tracking_data import (
    export_robmots_data,
    set_up_synth_track_errors,
    read_lineage_file,
)


def calculate_run_summary(track_error_folder: Path, percentages: Iterable, n_runs: int):
    """Combines the metric results of every run to a combined file and calculates the mean for each metric."""
    runs = range(n_runs)
    data = {}
    for p in percentages:
        data_path = (
            track_error_folder / f"percentage_{p}/run_{runs[0]}/metric_results.txt"
        )
        all_runs_df = pd.read_csv(data_path, sep=" ")
        for r in runs[1:]:
            data_path = (
                track_error_folder / f"percentage_{p}/run_{r}/metric_results.txt"
            )
            df = pd.read_csv(data_path, sep=" ")
            all_runs_df = pd.concat([all_runs_df, df])

        data[
            f"{track_error_folder.name}_percentage_{p}"
        ] = all_runs_df.mean()  # df.iloc[0].to_dict()
        data[f"{track_error_folder.name}_percentage_{p}_all"] = all_runs_df.to_dict(
            "list"
        )
        all_runs_df.to_csv(
            track_error_folder / f"percentage_{p}/all_runs_results.txt",
            sep=" ",
            index=None,
        )

    return data


@timing
def synth_tracks_main():
    ALL_DATA_SETS = ["Fluo-N2DH-SIM+"]
    CREATE_DATA = True
    CALC_MOT = True
    CALC_AOGM = True
    CREATE_THESIS_PLOTS = True
    CREATE_TRACKEVAL_PLOTS = False
    PERCENTAGES = [1, 2, 5, 10, 20]
    N_RUNS = 5
    EVAL_KEY = None  # None or could be "02" or "01" for specific dataset evaluation, or even parts of a path
    SAVE_PATH = (
        get_results_path() / "synth_tracking_errors"
    )  # this needs to be the folder of the data

    np.random.seed(2402)
    for p in ALL_DATA_SETS:
        if CREATE_DATA:
            data_path = Path(get_data_path() / p)
            data_set = data_path.name
            res_path = SAVE_PATH / data_set

            # Save the ground truth in the RobMots format
            for i in range(1, 3):
                gt_path = Path(get_data_path() / (f"{data_set}/{str(i).zfill(2)}_GT"))
                if not gt_path.is_dir():
                    continue

                segm_masks, img_size = extract_segmentation_data(gt_path / "TRA")
                lineage_df = read_lineage_file(gt_path / "TRA" / "man_track.txt")
                export_robmots_data(segm_masks, lineage_df, img_size, gt_path, gt=True)

            # Create the synthetically degraded tracking result data set
            set_up_synth_track_errors(data_path, res_path, PERCENTAGES, N_RUNS)

        if CALC_MOT or CALC_AOGM:
            eval_paths = collect_leaf_paths(SAVE_PATH / p)
            for data_path in eval_paths:
                path_str = str(data_path)
                if (
                    "data" not in path_str
                    or "avg_results" in path_str
                    or "plots" in path_str
                ):
                    continue
                if EVAL_KEY and EVAL_KEY not in path_str:
                    continue

                path_parts = data_path.relative_to(SAVE_PATH).parts
                name_data_set = path_parts[0]
                id_data_set = path_parts[1]

                current_run_folder = SAVE_PATH / "\\".join(path_parts[:5])

                # Check if the folder was already evaluated
                if len(list(Path(current_run_folder).rglob("metric_results.txt"))) > 0:
                    continue

                print(f'\n\n{100*"#"}\n')
                print(path_parts[:5])

                trackeval_all_results, aogm_all_results = pd.DataFrame(), pd.DataFrame()
                gt_path = get_data_path() / name_data_set / (id_data_set + "_GT")
                if CALC_MOT:
                    res_path = current_run_folder / "trackers"

                    # Omit the extensive output of the trackeval package and only display HOTA metric results.
                    old_stdout = sys.stdout  # backup current stdout
                    sys.stdout = open(os.devnull, "w")
                    trackeval_success = calculate_trackeval_scores(
                        gt_path / "gt", res_path, print_res=False
                    )
                    sys.stdout = old_stdout  # reset old stdout

                    path = f"{data_path.parent.parent}\\results\\bdd_mots\\person_summary.txt"
                    trackeval_all_results = pd.read_csv(path, sep=" ")
                    print(f' HOTA measure: {trackeval_all_results["HOTA"][0]/100:.6f}')

                if CALC_AOGM:
                    aogm_all_results = pd.DataFrame()

                    aogm_results = calc_ctc_scores(current_run_folder, gt_path)
                    aogm_error_counts = extract_aogm_info(
                        current_run_folder / "TRA_log.txt"
                    )["counts"]
                    aogm_all_results = {**aogm_results, **aogm_error_counts}
                    aogm_all_results = pd.DataFrame(
                        [aogm_all_results], columns=aogm_all_results.keys()
                    )
                    aogm_all_results[["DET", "SEG", "TRA"]] = (
                        aogm_all_results[["DET", "SEG", "TRA"]] * 100
                    )

                if CALC_AOGM or CALC_MOT:
                    combined_all_results = pd.DataFrame()
                    if not aogm_all_results.empty:
                        combined_all_results = pd.concat(
                            [combined_all_results, aogm_all_results], axis=1
                        )

                    if not trackeval_all_results.empty:
                        combined_all_results = pd.concat(
                            [combined_all_results, trackeval_all_results], axis=1
                        )

                    combined_all_results.to_csv(
                        f"{current_run_folder}\\metric_results.txt",
                        index=False,
                        sep=" ",
                    )
        for i in range(1, 3):
            if CALC_MOT or CALC_AOGM:
                track_error_paths_list = [
                    f
                    for f in ((SAVE_PATH / (f"{p}/{str(i).zfill(2)}"))).iterdir()
                    if f.is_dir()
                ]
                if EVAL_KEY:
                    track_error_paths_list = [
                        f
                        for f in track_error_paths_list
                        if EVAL_KEY in str(f) and "plots" not in str(f)
                    ]
                for folder_path in track_error_paths_list[:]:
                    if "plots" in str(folder_path):
                        continue
                    data = calculate_run_summary(folder_path, PERCENTAGES, N_RUNS)
                    if CREATE_TRACKEVAL_PLOTS:
                        plot_trackeval_results(data, folder_path, PERCENTAGES, N_RUNS)

            if CREATE_THESIS_PLOTS:
                create_all_plots(
                    Path(SAVE_PATH / (f"{p}/{str(i).zfill(2)}")), save=True, show=True
                )


if __name__ == "__main__":
    synth_tracks_main()
