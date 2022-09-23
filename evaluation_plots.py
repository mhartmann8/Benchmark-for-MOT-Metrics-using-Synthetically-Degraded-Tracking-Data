"""Methods to generate metric summary and comparison plots."""
import os
import pandas as pd
from pathlib import Path
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import seaborn as sns
import trackeval

sns.set_style("white")
sns.set_context("talk")

MEDIUM_SIZE = 28
BIGGER_SIZE = 30

plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
plt.rcParams["xtick.labelsize"] = 23
plt.rcParams["ytick.labelsize"] = 23
plt.rc("axes", titlesize=21)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("legend", fontsize=23)  # legend fontsize
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["axes.edgecolor"] = "black"

LEGEND_LABEL_DICT = {
    # key : [TYPE, SPECIAL]
    "fragmentation_noFragLen_noParent": ["Fragmentation", "No Predecessor"],
    "fragmentation_noFragLen_setParent": ["Fragmentation", "Set Predecessor"],
    "id_switch": ["ID Switch", ""],
    "missing_daughters_frames_keepInheritence": [
        "Missing First Daughter Frames",
        "Set Predecessor",
    ],
    "missing_daughters_frames_noInheritence": [
        "Missing First Daughter Frames",
        "No Predecessor",
    ],
    "missing_last_parent_frame_keepInheritence": [
        "Missing Last Parent Frame",
        "Set Predecessor",
    ],
    "missing_last_parent_frame_noInheritence": [
        "Missing Last Parent Frame",
        "No Predecessor",
    ],
    "mixed_track_errors_setParent": ["Mixed Error", "Set Predecessor"],
    "mixed_track_errors_noParent": ["Mixed Error", "No Predecessor"],
    "missing_single_daughter_frame_keepInheritence": [
        "Missing Single Daughter Frame",
        "Set Predecessor",
    ],
    "missing_single_daughter_frame_noInheritence": [
        "Missing Single Daughter Frame",
        "No Predecessor",
    ],
    "fragmentation_5FragLen_setParent": ["Fragmentation FragLen 5", "Set Predecessor"],
    "fragmentation_25FragLen_setParent": [
        "Fragmentation FragLen 25",
        "Set Predecessor",
    ],
    "fragmentation_50FragLen_setParent": [
        "Fragmentation FragLen 50",
        "Set Predecessor",
    ],
    "fragmentation_5FragLen_noParent": ["Fragmentation FragLen 5", "No Predecessor"],
    "fragmentation_25FragLen_noParent": ["Fragmentation FragLen 25", "No Predecessor"],
    "fragmentation_50FragLen_noParent": ["Fragmentation FragLen 50", "No Predecessor"],
    "missing_single_mitosis_link": ["Missing Single Mitosis Link", ""],
    "noMitosis": ["No Mitosis Links", ""],
    "fragmentation_nPercentFragLen_setParent": [
        "Fragmentation n% FragLen",
        "Set Predecessor",
    ],
    "fragmentation_nPercentFragLen_noParent": [
        "Fragmentation n% FragLen",
        "No Predecessor",
    ],
}
markers = [
    "s",
    "d",
    "^",
    "p",
    "h",
    "8",
    "v",
    "X",
    "D",
    "8",
    "o",
    "P",
    ".",
    "1",
    "2",
    mpl.markers.CARETDOWNBASE,
    mpl.markers.CARETUPBASE,
    "<",
]
# MARKER_DICT = {kvals[0]: markers[i] for i, kvals in enumerate(LEGEND_LABEL_DICT.items()) if i < len(markers)}
COLOR_PALETTE = [
    "#e69f00",
    "#56b4e9",
    "#009e73",
    "#f0e442",
    "#0072b2",
    "#d55e00",
    "#cc79a7",
    "#000000",
]


def trackeval_plots(
    data,
    percentages,
    out_loc,
    y_label,
    x_label,
    sort_label,
    bg_label=None,
    bg_function=None,
    set_title_to=None,
    settings=None,
):
    """Creates a scatter plot comparing multiple trackers between two metric fields, with one on the x-axis and the
    other on the y axis. Adds pareto optical lines and (optionally) a background contour.
    Modified method originally from: (plotting.py) of https://github.com/JonathonLuiten/TrackEval
    Inputs:
        data: dict of dicts such that data[tracker_name][metric_field_name] = float
        y_label: the metric_field_name to be plotted on the y-axis
        x_label: the metric_field_name to be plotted on the x-axis
        sort_label: the metric_field_name by which trackers are ordered and ranked
        bg_label: the metric_field_name by which (optional) background contours are plotted
        bg_function: the (optional) function bg_function(x,y) which converts the x_label / y_label values into bg_label.
        settings: dict of plot settings with keys:
            'gap_val': gap between axis ticks and bg curves.
            'num_to_plot': maximum number of trackers to plot
    """

    # Only loaded when run to reduce minimum requirements
    from matplotlib import pyplot as plt

    # Get plot settings
    if settings is None:
        gap_val = 10
        num_to_plot = 20
    else:
        gap_val = settings["gap_val"]
        num_to_plot = settings["num_to_plot"]

    if (bg_label is None) != (bg_function is None):
        raise trackeval.plotting.TrackEvalException(
            "bg_function and bg_label must either be both given or neither given."
        )

    # Extract data
    tracker_names = np.array([t for t in data.keys() if not "all" in t])
    sort_index = np.array([data[t][sort_label] for t in tracker_names]).argsort()[::-1]
    x_values = np.array([data[t][x_label] for t in tracker_names])[sort_index][
        :num_to_plot
    ]
    y_values = np.array([data[t][y_label] for t in tracker_names])[sort_index][
        :num_to_plot
    ]

    # Print info on what is being plotted
    tracker_names = tracker_names[sort_index][:num_to_plot]
    print(
        "\nPlotting %s vs %s, for the following (ordered) trackers:"
        % (y_label, x_label)
    )
    for i, name in enumerate(tracker_names):
        print("%i: %s" % (i + 1, name))

    fig, ax = plt.subplots(figsize=(8, 8))

    colwheel = iter(
        [
            "#e69f00",
            "#56b4e9",
            "#009e73",
            "#f0e442",
            "#0072b2",
            "#d55e00",
            "#cc79a7",
            "#000000",
        ]
    )
    minx, maxx, miny, maxy = (
        np.min(x_values),
        np.max(x_values),
        np.min(y_values),
        np.max(y_values),
    )
    for i, t in enumerate(tracker_names):
        plt.plot(
            data[f"{t}_all"][x_label],
            data[f"{t}_all"][y_label],
            color=next(colwheel),
            linestyle="none",
            alpha=0.6,
            label=f'{t.split("_")[-1]} %',
            marker=markers[i],
            markersize=10,
        )
        minx = min(minx, np.min(data[f"{t}_all"][x_label]))
        maxx = max(maxx, np.max(data[f"{t}_all"][x_label]))
        miny = min(miny, np.min(data[f"{t}_all"][y_label]))
        maxy = max(maxy, np.max(data[f"{t}_all"][y_label]))

    # Plot data points with number labels
    # labels = percentages#np.arange(len(y_values)) + 1
    plt.plot(
        x_values, y_values, color="black", linestyle="None", marker="+", markersize=16
    )  # linestyle=(0, (1, 10)),
    # for xx, yy, l in zip(x_values, y_values, labels):
    #     plt.text(xx, yy, f'{str(l)} %', color="red", fontsize=15)

    # Plot pareto optimal lines
    # trackeval.plotting._plot_pareto_optimal_lines(x_values, y_values)

    # Find best fitting boundaries for data
    # boundaries = trackeval.plotting._get_boundaries(x_values, y_values, round_val=gap_val/2)
    boundaries = trackeval.plotting._get_boundaries(
        np.array([minx, maxx]), np.array([miny, maxy]), round_val=gap_val / 2
    )

    # Plot background contour
    if bg_function is not None:
        trackeval.plotting._plot_bg_contour(bg_function, boundaries, gap_val)

    # Add extra explanatory text to plots
    # plt.text(0, -0.11, 'label order:\nHOTA', horizontalalignment='left', verticalalignment='center',
    #          transform=fig.axes[0].transAxes, color="red", fontsize=12)
    if bg_label is not None:
        plt.text(
            1,
            -0.11,
            "curve values:\n" + bg_label,
            horizontalalignment="right",
            verticalalignment="center",
            transform=fig.axes[0].transAxes,
            color="grey",
            fontsize=12,
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    title = y_label + " vs " + x_label
    if bg_label is not None:
        title += " (" + bg_label + ")"

    if not set_title_to:
        plt.title(title)
    else:

        plt.title(" ".join(LEGEND_LABEL_DICT[set_title_to]))

    plt.xticks(np.arange(0, 101, gap_val))
    plt.yticks(np.arange(0, 101, gap_val))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.tick_params(
        axis="both",
        direction="in",
        which="minor",
        length=5,
        bottom=True,
        top=False,
        left=True,
        right=False,
    )
    ax.tick_params(
        axis="both",
        direction="in",
        which="major",
        length=8,
        bottom=True,
        top=False,
        left=True,
        right=False,
    )
    plt.grid(True)
    min_x, max_x, min_y, max_y = boundaries

    plt.xlim(min_x, 100)  # max_x + 1)
    plt.ylim(min_y, 100)  # max_y + 1)

    plt.legend(
        loc="lower left", title="Error Percentages"
    )  # fontsize="small",  title_fontsize="small",
    plt.tight_layout()

    os.makedirs(out_loc, exist_ok=True)
    filename = os.path.join(out_loc, title.replace(" ", "_"))
    plt.savefig(filename + ".pdf", bbox_inches="tight")  # , pad_inches=0.05)
    plt.savefig(filename + ".png", bbox_inches="tight")  # , pad_inches=0.05)
    plt.close()


def plot_trackeval_results(
    data: dict, track_error_folder: Path, percentages, n_runs: int
):
    """Generate metric comparison plots in the trackeval format. Added the AOGM metrics to the plot list."""
    plots_list = trackeval.plotting.get_default_plots_list()
    plots_list.append(["TRA", "SEG", "TRA", None, None])
    plots_list.append(["SEG", "DET", "SEG", None, None])
    plots_list.append(["HOTA", "TRA", "TRA", None, None])
    plots_list.append(["IDF1", "TRA", "TRA", None, None])
    plots_list.append(["MOTA", "TRA", "TRA", None, None])

    title = track_error_folder.name
    print(title)
    out_path = track_error_folder / r"avg_results"
    os.makedirs(out_path, exist_ok=True)

    for args in plots_list:
        trackeval_plots(
            data,
            percentages,
            out_path,
            y_label=args[0],
            x_label=args[1],
            sort_label=args[2],
            bg_label=args[3],
            bg_function=args[4],
            set_title_to=title,
        )


def read_metric_results(fol):
    # Format: results_dict[track_error_folder:str][percentage:int][Metric Name:str]
    results_dict = {}
    for d_fol in fol.iterdir():
        for f in d_fol.iterdir():
            metric_result_path = f / "all_runs_results.txt"

            if "avg" in str(f) or not metric_result_path.is_file():
                continue
            results_dict.setdefault(f.parent.name, {})
            results_dict[f.parent.name][int(f.name.split("_")[1])] = pd.read_csv(
                metric_result_path, sep=" "
            )
    return results_dict


def subfig_evaluation_plot(
    data,
    metric_name,
    only_avg=False,
    linestyle="",
    linestyle_runs="",
    sort_legend=True,
    ax_limits=None,
    subplot_num=3,
):
    """
    Tracking Error Comparison per Metric, x: Percentage, y: Metric, legend: Tracking Errors
    Args:
        data: dict([track_error_folder:str][percentage:int][Metric Name:str])

    """
    plt.rc("legend", fontsize=19)  # legend fontsize

    # plt.rcParams.update({'font.size':14})
    x_label = "Fraction Tracking Errors (%)"
    fig, ax = plt.subplots(
        1,
        subplot_num,
        figsize=(subplot_num * 5.5, 8),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    colorwheel = iter(COLOR_PALETTE)

    marker_dist = 0.2
    totalmin = 0.99999
    totalmaxorder = []
    j = 0
    for i, t in enumerate(data.keys()):
        xs = sorted(data[t].keys())
        x_values = [
            x + len(data.keys()) * x * marker_dist + (i % 2) * 4 * marker_dist
            for x in range(len(xs))
        ]
        color = next(colorwheel)

        ys = [data[t][x][metric_name] for x in xs]
        if not only_avg:
            newxs = []
            newys = []
            for ix, x in enumerate(xs):
                val = data[t][x][metric_name]
                newxs.append(x_values[ix])
                newys.append(val / 100)
            ax[j].plot(
                newxs,
                newys,
                alpha=0.2,
                color=color,
                marker=markers[i],
                markersize=10,
                linestyle=linestyle_runs,
            )  # mfc="white")

        y_values = [p.mean() / 100 for p in ys]
        ax[j].plot(
            x_values,
            y_values,
            color=color,
            marker=markers[i],
            label=LEGEND_LABEL_DICT[t][1],
            markersize=15,
            linestyle=linestyle,
        )

        totalmin = min(totalmin, np.min(ys) / 100)
        totalmaxorder.append((i, np.sum(ys)))

        ax[j].set_title(LEGEND_LABEL_DICT[t][0], fontsize="x-small")
        ax[j].legend(
            loc="lower left", title="Tracking Errors"
        )  # fontsize="small",  title_fontsize="small"
        ax[j].get_legend()._legend_box.align = "left"
        j += i % 2

    # Labels & Legend
    if subplot_num == 2:
        # ax[0].set_xlabel(x_label)
        # ax[1].set_xlabel(x_label)
        fig.supxlabel(x_label, x=0.55, fontsize=MEDIUM_SIZE)
    else:
        ax[1].set_xlabel(x_label)
    ax[0].set_ylabel(metric_name + " Score")

    # Y Axis
    floortotalmin = totalmin // 0.01 / 100
    print(totalmin)

    for i in range(subplot_num):
        ax[i].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[i].grid(which="minor", color="#CCCCCC", linestyle="-", axis="x")
        ax[i].tick_params(
            axis="x",
            direction="in",
            which="minor",
            length=8,
            bottom=True,
            top=False,
            left=True,
            right=False,
        )

        if ax_limits:
            ax[i].set_ylim(ax_limits[0], ax_limits[1])
            ax[i].yaxis.set_major_locator(MultipleLocator(base=ax_limits[2]))
        ax[i].yaxis.set_minor_locator(AutoMinorLocator(3))
        ax[i].grid(axis="y", which="major", color="#CCCCCC", linestyle="-")
        ax[i].tick_params(
            axis="y",
            direction="in",
            which="major",
            length=8,
            bottom=True,
            top=False,
            left=True,
            right=False,
        )
        ax[i].tick_params(
            axis="y",
            direction="in",
            which="minor",
            length=5,
            bottom=True,
            top=False,
            left=True,
            right=False,
        )

    # X Axis
    modded_xticks = (
        np.array(x_values)
        - (marker_dist * 0.5 * (len(data.keys())))
        + (marker_dist * 0.5)
    )
    plt.xticks(modded_xticks, labels=xs)
    # plt.tight_layout()


def evaluation_plot(
    data,
    metric_name,
    only_avg=False,
    linestyle="",
    linestyle_runs="",
    sort_legend=True,
    ax_limits=None,
    short_legend=False,
):
    """
    Tracking Error Comparison per Metric, x: Percentage, y: Metric, legend: Tracking Errors
    Args:
        data: dict([track_error_folder:str][percentage:int][Metric Name:str])

    """
    plt.rc("legend", fontsize=23)  # legend fontsize

    # plt.rcParams.update({'font.size':14})
    x_label = "Fraction Tracking Errors (%)"
    fig, ax = plt.subplots(figsize=(10, 8))
    colorwheel = iter(COLOR_PALETTE)

    offset = 0.2
    totalmin = 0.9
    totalmaxorder = []
    for j, t in enumerate(data.keys()):
        xs = sorted(data[t].keys())
        x_values = [
            x + len(data.keys()) * x * offset + j * offset * 1.2 for x in range(len(xs))
        ]
        color = next(colorwheel)
        marker = markers[j]
        ys = [data[t][x][metric_name] for x in xs]
        if not only_avg:
            newxs = []
            newys = []
            for ix, x in enumerate(xs):
                val = data[t][x][metric_name]
                newxs.append(x_values[ix])
                newys.append(val / 100)
            plt.plot(
                newxs,
                newys,
                alpha=0.2,
                color=color,
                marker=marker,
                markersize=10,
                linestyle=linestyle_runs,
            )  # mfc="white") MARKER_DICT[t]

        y_values = [p.mean() / 100 for p in ys]
        if short_legend:
            legend_label = LEGEND_LABEL_DICT[t][0]
        else:
            legend_label = " ".join(LEGEND_LABEL_DICT[t])

        plt.plot(
            x_values,
            y_values,
            color=color,
            marker=marker,
            label=legend_label,
            markersize=15,
            linestyle=linestyle,
        )

        totalmin = min(totalmin, np.min(ys))
        totalmaxorder.append((j, np.sum(ys)))

    # Labels & Legend
    plt.xlabel(x_label)
    plt.ylabel(metric_name + " Score")

    if sort_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [
            tup[0]
            for tup in sorted(totalmaxorder, key=lambda tup: tup[1], reverse=True)
        ]
        legend = plt.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="lower left",
            title="Tracking Errors",
        )
        legend._legend_box.align = "left"
    else:
        legend = plt.legend(loc="lower left", title="Tracking Errors")
        legend._legend_box.align = "left"

    if ax_limits:
        plt.ylim(ax_limits[0], ax_limits[1])
        ax.yaxis.set_major_locator(MultipleLocator(base=ax_limits[2]))
    plt.tick_params(
        axis="y",
        direction="in",
        which="major",
        length=8,
        bottom=True,
        top=False,
        left=True,
        right=False,
    )
    plt.tick_params(
        axis="y",
        direction="in",
        which="minor",
        length=5,
        bottom=True,
        top=False,
        left=True,
        right=False,
    )
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(axis="y", which="major", color="#CCCCCC", linestyle="-")

    # X Axis
    # plt.tick_params(axis="x", direction='out',which='major', length=8, bottom=True, top=False, left=True, right=False)
    plt.tick_params(
        axis="x",
        direction="in",
        which="minor",
        length=5,
        bottom=True,
        top=False,
        left=True,
        right=False,
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.xticks(
        np.array(x_values) - (offset * 0.5 * (len(data.keys()))) + (offset * 0.5),
        labels=xs,
    )
    ax.grid(which="minor", color="#CCCCCC", linestyle="-", axis="x")

    # plt.gca().set_position([0, 0, 1, 1])
    plt.tight_layout()


def create_all_plots(results_path, save=False, show=True):
    # selects the graphs that should be plotted
    # dict(graph_name : [method_name, [metric_names], (ymin, ymax, majorlocatorbase), [list of keys], subplot_num)
    plot_selection_config = {
        "Mitosis": [
            "subfig_evaluation_plot",
            ["TRA"],
            (0.9925, 1.0001, 0.002),
            [
                "missing_daughters_frames_noInheritence",
                "missing_daughters_frames_keepInheritence",
                "missing_last_parent_frame_noInheritence",
                "missing_last_parent_frame_keepInheritence",
                "missing_single_daughter_frame_noInheritence",
                "missing_single_daughter_frame_keepInheritence",
            ],
            3,
        ],
        "FragAndMixed": [
            "subfig_evaluation_plot",
            ["TRA"],
            (0.71, 1.0001, 0.05),
            [
                "fragmentation_noFragLen_noParent",
                "fragmentation_noFragLen_setParent",
                "mixed_track_errors_noParent",
                "mixed_track_errors_setParent",
            ],
            2,
        ],
        "MixedErrors": [
            "evaluation_plot",
            ["TRA", "HOTA", "IDF1", "MOTA"],
            (0.53, 1.0, 0.05),
            [
                "fragmentation_noFragLen_setParent",
                "missing_last_parent_frame_keepInheritence",
                "missing_daughters_frames_keepInheritence",
                "id_switch",
                "mixed_track_errors_setParent",
            ],
            None,
            True,
        ],
        "FragLenVariation": [
            "evaluation_plot",
            ["TRA", "HOTA"],
            (0.7, 1.0, 0.05),
            [
                "fragmentation_5FragLen_noParent",
                "fragmentation_25FragLen_noParent",
                "fragmentation_50FragLen_noParent",
            ],
            3,
        ],
        "FragAndID": [
            "evaluation_plot",
            ["TRA", "HOTA", "IDF1", "MOTA"],
            (0.53, 1.0, 0.05),
            [
                "fragmentation_noFragLen_noParent",
                "fragmentation_noFragLen_setParent",
                "id_switch",
            ],
        ],
        # "MitosisExtra":["evaluation_plot", ["TRA"], (0.9925, 1.0, 0.002), ['missing_single_mitosis_link', 'noMitosis']],
    }

    results_dict = read_metric_results(results_path)
    output_path = Path(f"{results_path}\\plots")
    os.makedirs(output_path, exist_ok=True)

    for name, args in plot_selection_config.items():
        for metric in args[1]:
            try:
                specific_results = {k: results_dict[k] for k in args[3]}
                print(metric, "\n", specific_results.keys())
            except:
                print(
                    f"Error in plotting {metric} results. Try computing the metric scoring first."
                )
                return
            if "subfig" in args[0]:
                eval(args[0])(
                    specific_results,
                    metric,
                    only_avg=False,
                    linestyle="",
                    linestyle_runs="",
                    sort_legend=False,
                    ax_limits=args[2],
                    subplot_num=args[4],
                )
            else:
                if len(args) > 5:
                    eval(args[0])(
                        specific_results,
                        metric,
                        only_avg=False,
                        linestyle="",
                        linestyle_runs="",
                        sort_legend=False,
                        ax_limits=args[2],
                        short_legend=args[5],
                    )
                else:
                    eval(args[0])(
                        specific_results,
                        metric,
                        only_avg=False,
                        linestyle="",
                        linestyle_runs="",
                        sort_legend=False,
                        ax_limits=args[2],
                    )

            if save:
                plt.savefig(
                    output_path / f"{name}_{metric}.pdf", dpi=1000, bbox_inches="tight"
                )
                plt.savefig(
                    output_path / f"{name}_{metric}.png", dpi=1000, bbox_inches="tight"
                )
            if show:
                plt.show()
