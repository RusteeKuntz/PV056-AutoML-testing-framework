import json
import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from pv056_2019.visualize.main_box import print_boxplots
from pv056_2019.schemas import GraphBoxStepSchema, GraphScatterStepSchema
from pv056_2019.utils import valid_path, convert_dict_to_parameter_pairs
from pv056_2019.visualize import SORT_FUNCTIONS, setup_arguments, prepare_data


def print_nice_scatterplot(data: pd.DataFrame,
                           graph_filename: str,
                           col_examined: str,
                           col_grouped_by: str,
                           col_related: str,
                           title: str,
                           legend_title: str,
                           x_title: str,
                           y_title: str,
                           max_y_val=None,
                           min_y_val=None,
                           convert_col_related_from_json=True):
    # preset configuration
    # scale=2
    max_marker_size = 1000  # *scale
    min_marker_size = 1  # *scale

    # ticks are calculated in respect to max and min value of the Y axis
    if max_y_val is None:
        max_y_val = data[col_examined].max()
    if min_y_val is None:
        min_y_val = data[col_examined].min()
    _y_tick = (max_y_val - min_y_val) / 40
    y_ticks = np.concatenate([np.arange(0, min_y_val - _y_tick, -_y_tick)[::-1], np.arange(0, max_y_val, _y_tick)])

    # here we transform data from the column containing the parameter we are investigating (col_related)
    # and read it as json, then extracting parameters to more readable string
    if convert_col_related_from_json:
        try:
            data[col_related] = data[col_related].map(convert_dict_to_parameter_pairs)
        except TypeError:
            print("Data are not interpretable as JSON.")

    # Scatterplot create figure
    # _fig = plt.figure( figsize=(8*scale,40*scale))
    _fig = plt.figure(figsize=(8, 40))

    # Create an axes instance
    ax1 = _fig.add_subplot(111)
    ax1.set_title(title,
                  fontsize=25  # *scale
                  )
    ax1.set_xlabel(x_title,
                   fontsize=25  # *scale
                   )
    ax1.set_ylabel(y_title,
                   rotation=90,
                   fontsize=25  # *scale
                   )
    # this sorts times and labels for display in the boxplot by the parameters of the boxplots
    # data_to_plot_arr, labels = zip(*sorted(zip(data_to_plot_arr,labels), key=lambda e: e[1] ))

    groups = []
    # get the dataframes with their group names into list
    for group, group_df in data.groupby(col_grouped_by):
        groups.append((group, group_df))

    # sort the list by the parameter so we can apply reasonable coloring
    groups = sorted(groups, key=lambda x: x[0])
    current_size = max_marker_size
    # use seaborn to generate list of enough colors from a color pallete - it is graded
    colors = sns.color_palette(sns.dark_palette('cyan', n_colors=len(groups)), n_colors=len(groups))
    for group, group_df in groups:
        # Create the scatterplot
        ax1.scatter(x=group_df[col_related], y=group_df[col_examined], label=str(group) + ' % ', color=colors.pop(),
                    s=current_size)
        current_size -= (max_marker_size - min_marker_size) / len(groups)

    # ax1.set_xticklabels(['1', '2', '5', '10', '50', '100', '500', '200'])
    ax1.set_yticks(y_ticks)
    ax1.tick_params(axis='x',
                    rotation=90,
                    labelsize=22  # *scale
                    )
    ax1.tick_params(axis='y',
                    labelsize=42  # *scale
                    )
    # ax1.grid(True)
    legend = plt.legend(loc="upper right",
                        bbox_to_anchor=(0, 0),
                        title=legend_title,
                        ncol=2,
                        prop={
                            'size': 16  # *scale
                        }
                        )
    legend.get_title().set_fontsize(22  # *scale
                                    )
    _fig.savefig(graph_filename, bbox_inches="tight", dpi=600)
    plt.close(_fig)


def main():
    parser = setup_arguments()
    args = parser.parse_args()
    with open(args.config_graph, "r") as conf_file:
        conf = GraphScatterStepSchema(**json.load(conf_file))

    df, out_fp = prepare_data(args, conf)

    if conf.separate_graphs_for_different_values_in_column is not None:
        gbc = df.groupby(conf.separate_graphs_for_different_values_in_column)
        for group, group_df in gbc:
            legend_appendix = " Data are selected for value" + (
                "s " if not isinstance(group, str) else " ") + group + " in column" + ("s " if not isinstance(group,
                                                                                                              str) else " ") + conf.separate_graphs_for_different_values_in_column + "."

            out_split = out_fp.split(".")
            if not isinstance(group, str):
                group_out_fp = ".".join(out_split[:-1]) + "_" + re.sub('[^\\w-_. ]', '_', ("_".join(group))) + out_split[
                    -1]
            else:
                group_out_fp = ".".join(out_split[:-1]) + "_" + re.sub('[^\\w-_. ]', '_', (group)) + out_split[-1]

            print_nice_scatterplot(data=group_df,
                                   graph_filename=group_out_fp,
                                   col_examined=conf.col_examined,
                                   col_grouped_by=conf.col_grouped_by,
                                   col_related=conf.col_related,
                                   title=conf.title,
                                   x_title=conf.x_title,
                                   y_title=conf.y_title,
                                   legend_title=conf.legend_title + legend_appendix,
                                   max_y_val=conf.max_y_val,
                                   min_y_val=conf.min_y_val,
                                   convert_col_related_from_json=conf.convert_col_related_from_json)

            # print_nice_scatterplot(data=group_df,
            #                        graph_filename=grouped_by + '_od_params/' + od_method_name + '-scatter-od_params-' + group + '.png',
            #                        col_examined="gain",
            #                        col_related="od_params",
            #                        col_grouped_by="removed",
            #                        x_title='n_neighbors value',
            #                        y_title='Increase in gain after OD',
            #                        title="Changes in accuracy " + title + " " + group + "\nbased on parameter n_neighbors \n of OD method " + od_method_name + " for different % of removed outliers\n",
            #                        legend_title="% of removed outliers")
    else:
        print_nice_scatterplot(data=df,
                               graph_filename=out_fp,
                               col_examined=conf.col_examined,
                               col_grouped_by=conf.col_grouped_by,
                               col_related=conf.col_related,
                               title=conf.title,
                               x_title=conf.x_title,
                               y_title=conf.y_title,
                               legend_title=conf.legend_title,
                               max_y_val=conf.max_y_val,
                               min_y_val=conf.min_y_val,
                               convert_col_related_from_json=conf.convert_col_related_from_json)
