import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pv056_2019.schemas import GraphBoxStepSchema
from pv056_2019.utils import valid_path, convert_dict_to_parameter_pairs
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
                   sort_func=None,
                   min_y_val=None,
                   max_y_val=None,
                   convert_col_related_from_json=True,
                   show_fliers: bool = True
                   ):
    # here we transform data from the column containing the parameter we are investigating (col_related)
    # and read it as json, then extracting parameters to more readable string
    if convert_col_related_from_json:
        try:
            data[col_related] = data[col_related].map(convert_dict_to_parameter_pairs)
        except TypeError:
            print("Data are not interpretable as JSON.")

    g = data.groupby([col_related])  # ["accuracy"].sum().reset_index()

    # graph parameters
    scale = 1
    mean_color='b'
    mean_marker='o'


    labels = []
    data_to_plot_arr = []
    #switch = True


    for group, group_df in g:
        data_to_plot_arr.append(group_df[col_examined])
        labels.append(group)

    # dynamically set parameters of the graphs so that they are uniform across all graphs, but are minimalised
    figsize = ((len(g)) * scale, 25 * scale)  # originally (60, 30)
    if max_y_val is None:
        max_y_val = data[col_examined].max()
    if min_y_val is None:
        min_y_val = data[col_examined].min()
    tick = (max_y_val - min_y_val) / 40
    y_labels = np.concatenate([np.arange(0, min_y_val - tick, -tick)[::-1], np.arange(0, max_y_val + 6 * tick, tick)])

    # Create a figure instance
    _fig = plt.figure( figsize=figsize)

    # Create an axes instance
    _ax = _fig.add_subplot(111)
    _ax.set_xlabel(col_related, fontsize=20*scale)
    if sort_func is not None:
        # this sorts times and labels for display in the boxplot by the parameters of the boxplots
        data_to_plot_arr, labels = zip(*sorted(zip(data_to_plot_arr,labels), key=sort_func ))

    # Create the boxplot

    bp = _ax.boxplot(data_to_plot_arr, positions=[x for x in range(len(labels))], showfliers=show_fliers)
    # following function is described here: https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
    _ax.plot([x for x in range(len(labels))], list(map(lambda x: x.mean(), list(data_to_plot_arr))), marker=mean_marker, color=mean_color)
    _ax.set_title(title,
                  fontsize=25 * scale)
    _ax.set_xlabel(x_title, fontsize=25 * scale)
    _ax.set_ylabel(y_title, rotation=90, fontsize=25 * scale)
    _ax.set_xticklabels(labels, rotation=90)
    _ax.set_yticks(y_labels)
    _ax.tick_params(axis='x', labelsize=22*scale)
    _ax.tick_params(axis='y', labelsize=22*scale)

    # custom legend elements gymnastics (it is really awful, but I coudl not find better solution)
    colors = [mean_color]
    sizes = [6*scale]
    texts = ["Mean"]
    patches = [plt.plot([], [], marker=mean_marker, ms=sizes[i], ls="", mec=None, color=colors[i],
                        label="{:s}".format(texts[i]))[0] for i in range(len(texts))]

    legend = plt.legend(handles=patches,
                        bbox_to_anchor=[0, 0],
                        loc='upper right',
                        title="Boxplots show first and third quartile,\n with variability represented with whiskers",
                        ncol=2,
                        prop={'size': 16 * scale})
    legend.get_title().set_fontsize(16 * scale)
    _ax.grid(True)

    # Save the figure
    _fig.savefig(graph_filename+'.png', bbox_inches='tight')
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
            legend_appendix = " Data are selected for value" + "s " if len(
                group) > 1 else " " + group + " in column" + "s " if len(
                group) > 1 else " " + conf.separate_graphs_for_different_values_in_column + "."
            raise NotImplementedError
    else:
        print_boxplots(data=df,
                   graph_filename=out_fp,
                   col_examined=conf.col_examined,
                   col_related=conf.col_related,
                   title=conf.title,
                   x_title=conf.x_title,
                   y_title=conf.x_title,
                   sort_func=SORT_FUNCTIONS[conf.sort_func_name],
                   min_y_val=conf.min_y_val,
                   max_y_val=conf.max_y_val,
                   convert_col_related_from_json=conf.convert_col_related_from_json,
                   show_fliers=conf.show_fliers)

