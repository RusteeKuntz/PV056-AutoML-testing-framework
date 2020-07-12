import json
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pv056_2019.schemas import GraphBoxStepSchema
from pv056_2019.utils import valid_path, convert_dict_to_parameter_pairs, extract_parameter_value_as_int
from pv056_2019.visualize import SORT_FUNCTIONS, setup_arguments, prepare_data

"""
argument sort_func is a comparator function applied to a tuple of two elements: (data series, name). It sorts data in the graph.
"""


def print_boxplots(data: pd.DataFrame,
                   graph_filename: str,
                   col_examined: str,
                   col_related: str,
                   title: str,
                   x_title: str,
                   y_title: str,
                   dpi: int,
                   height_multiplier: float = 1,
                   width_multiplier: float = 1,
                   sort_func=None,
                   min_y_val=None,
                   max_y_val=None,
                   extract_col_related=None,
                   convert_col_related_from_json=True,
                   show_fliers: bool = True
                   ):
    if convert_col_related_from_json and extract_col_related is None:
        try:
            if isinstance(col_related, List):
                for col in col_related:
                    data[col] = data[col].map(convert_dict_to_parameter_pairs)
            else:
                data[col_related] = data[col_related].map(convert_dict_to_parameter_pairs)
        except TypeError:
            print("Data are not interpretable as JSON.")

    if isinstance(col_related, List):
        new_col_related = "_".join(col_related)
        data[new_col_related] = data[col_related[0]].astype(str)
        for col in col_related[1:]:
            data[new_col_related] = data[new_col_related] + "_" + data[col]

        # data[].astype(str) + '_' + big['foo'] + '_' + big['new']
    else:
        new_col_related = col_related

    # now, after the related columns are joined into one, we can extract parameters
    if extract_col_related is not None:
        data[new_col_related] = data[new_col_related].map(
            lambda x: extract_parameter_value_as_int(x, extract_col_related))

    g = data.groupby([col_related])  # ["accuracy"].sum().reset_index()

    # graph parameters
    # scale = 1
    mean_color = 'b'
    mean_marker = 'o'

    labels = []
    data_to_plot_arr = []
    # switch = True

    for group, group_df in g:
        data_to_plot_arr.append(group_df[col_examined])
        labels.append(group)

    # dynamically set parameters of the graphs so that they are uniform across all graphs, but are minimalised
    # figsize = ((len(g)) * scale, 25 * scale)  # originally (60, 30)
    figsize = ((len(g)*width_multiplier), 30*height_multiplier)  # originally (60, 30)
    if max_y_val is None:
        max_y_val = data[col_examined].max()
    if min_y_val is None:
        min_y_val = data[col_examined].min()
    _y_tick = (max_y_val - min_y_val) / 40
    if min_y_val < 0:
        y_ticks = np.concatenate([np.arange(0, min_y_val - _y_tick, -_y_tick)[::-1], np.arange(0, max_y_val, _y_tick)])
    else:
        y_ticks = [np.arange(min_y_val, max_y_val, _y_tick)]
    # Create a figure instance
    _fig = plt.figure(figsize=figsize)

    # Create an axes instance
    _ax = _fig.add_subplot(111)
    _ax.set_xlabel(col_related, fontsize=20)  # *scale)
    if sort_func is not None:
        # this sorts times and labels for display in the boxplot by the parameters of the boxplots
        data_to_plot_arr, labels = zip(*sorted(zip(data_to_plot_arr, labels), key=sort_func))

    # Create the boxplot

    bp = _ax.boxplot(data_to_plot_arr, positions=[x for x in range(len(labels))], showfliers=show_fliers)
    # following function is described here: https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
    _ax.plot([x for x in range(len(labels))], list(map(lambda x: x.mean(), list(data_to_plot_arr))), marker=mean_marker,
             color=mean_color)
    _ax.set_title(title,
                  fontsize=25  # * scale
                  )
    _ax.set_xlabel(x_title, fontsize=25  # * scale
                   )
    _ax.set_ylabel(y_title, rotation=90, fontsize=25  # * scale
                   )
    _ax.set_xticklabels(labels, rotation=90)
    _ax.set_yticks(y_ticks)
    _ax.tick_params(axis='x', labelsize=22  # *scale
                    )
    _ax.tick_params(axis='y', labelsize=22  # *scale
                    )

    # custom legend elements gymnastics (it is really awful, but I coudl not find better solution)
    colors = [mean_color]
    # sizes = [6*scale]
    sizes = [6]
    texts = ["Mean"]
    patches = [plt.plot([], [], marker=mean_marker, ms=sizes[i], ls="", mec=None, color=colors[i],
                        label="{:s}".format(texts[i]))[0] for i in range(len(texts))]

    legend = plt.legend(handles=patches,
                        bbox_to_anchor=[0, 0],
                        loc='upper right',
                        title="Boxplots show first and third quartile,\n with variability represented with whiskers",
                        ncol=2,
                        # prop={'size': 16 * scale})
                        prop={'size': 16})

    legend.get_title().set_fontsize(16  # * scale
                                    )
    _ax.grid(True)

    # Save the figure
    _fig.savefig(graph_filename + '.png', bbox_inches='tight', dpi=dpi)
    plt.close(_fig)


def main():
    parser = setup_arguments()
    args = parser.parse_args()
    with open(args.config_graph, "r") as conf_file:
        conf = GraphBoxStepSchema(**json.load(conf_file))

    df, out_fp = prepare_data(args, conf)

    if conf.separate_graphs_for_different_values_in_column is not None:
        # extract data

        gbc = df.groupby(conf.separate_graphs_for_different_values_in_column)
        for group, group_df in gbc:
            legend_appendix = " Data are selected for value" + (
                "s " if not isinstance(group, str) else " ") + group + " in column" + ("s " if not isinstance(group,
                                                                                                              str) else " ") + conf.separate_graphs_for_different_values_in_column + "."
            raise NotImplementedError
    else:
        print_boxplots(data=df,
                       graph_filename=out_fp,
                       col_examined=conf.col_examined,
                       col_related=conf.col_related,
                       title=conf.title,
                       x_title=conf.x_title,
                       y_title=conf.x_title,
                       dpi=conf.dpi,
                       sort_func=SORT_FUNCTIONS[conf.sort_func_name],
                       min_y_val=conf.min_y_val,
                       max_y_val=conf.max_y_val,
                       extract_col_related=conf.extract_col_related,
                       convert_col_related_from_json=conf.convert_col_related_from_json,
                       show_fliers=conf.show_fliers
                       )
