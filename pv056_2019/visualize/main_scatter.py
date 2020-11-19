import json
import re
from typing import Union, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


from pv056_2019.schemas import GraphBoxStepSchema, GraphScatterStepSchema
from pv056_2019.utils import valid_path, convert_dict_to_parameter_pairs, extract_parameter_value_as_int, \
    comp_str_as_num
from pv056_2019.visualize import setup_arguments, prepare_data


def print_nice_scatterplot(data: pd.DataFrame,
                           graph_filename: str,
                           col_examined: str,
                           col_grouped_by: Union[str, List[str]],
                           col_related: Union[str, List[str]],
                           title: str,
                           legend_title: str,
                           x_title: str,
                           y_title: str,
                           dpi: int,
                           height_multiplier: float = 1,
                           width_multiplier: float = 1,
                           max_y_val=None,
                           min_y_val=None,
                           convert_col_related_from_json=False,
                           convert_col_grouped_by_from_json=False,
                           extract_col_related=None,
                           extract_col_grouped_by=None
                           ):
    # preset configuration
    # scale=2
    height = 30
    max_marker_size = 1000  # *scale
    min_marker_size = 1  # *scale
    # TODO: Allow zooming
    # ticks are calculated in respect to max and min value of the Y axis
    if max_y_val is None:
        max_y_val = data[col_examined].max()
    if min_y_val is None:
        min_y_val = data[col_examined].min()
    _y_tick = (max_y_val - min_y_val) / (height / 2)

    # if min_y_val == max_y_val:
    #     y_ticks = np.array([min_y_val])
    # else:
    #     if min_y_val < 0:
    #         y_ticks = np.concatenate([np.arange(0, min_y_val - _y_tick, -_y_tick)[::-1], np.arange(0, max_y_val + _y_tick, _y_tick)])
    #     else:
    #         y_ticks = np.arange(min_y_val, max_y_val, _y_tick)

    print("Ticks ok")
    # here we transform data from the column containing the parameter we are investigating (col_related)
    # and read it as json, then extracting parameters to more readable string

    if convert_col_related_from_json and extract_col_related is None:
        print("Converting from json", flush=True)

        if isinstance(col_related, List):
            for col in col_related:
                try:
                    data[col] = data[col].map(convert_dict_to_parameter_pairs)
                except:
                    print("Data are not interpretable as JSON.")
        else:
            try:
                data[col_related] = data[col_related].map(convert_dict_to_parameter_pairs)
            except:
                print("Data are not interpretable as JSON.")

    if convert_col_grouped_by_from_json and extract_col_grouped_by is None:
        print("Converting from json", flush=True)

        if isinstance(col_grouped_by, List):
            for col in col_grouped_by:
                try:
                    data[col] = data[col].map(convert_dict_to_parameter_pairs)
                except:
                    print("Data are not interpretable as JSON.")
        else:
            try:
                data[col_grouped_by] = data[col_grouped_by].map(convert_dict_to_parameter_pairs)
            except:
                print("Data are not interpretable as JSON.")


    if isinstance(col_related, List):
        new_col_related = "_".join(col_related)
        data[new_col_related] = data[col_related[0]].astype(str)
        for col in col_related[1:]:
            data[new_col_related] = data[new_col_related] + ", " + data[col]

        #data[].astype(str) + '_' + big['foo'] + '_' + big['new']
    else:
        new_col_related = col_related

    # now, after the related columns are joined into one, we can extract parameters
    if extract_col_related is not None:
        print("Extracting col_related", flush=True)
        data[new_col_related] = data[new_col_related].map(lambda x: extract_parameter_value_as_int(x, extract_col_related))
    if extract_col_grouped_by is not None:
        print("Extracting col_grouped_by", flush=True)
        if isinstance(col_grouped_by, List):
            for group in col_grouped_by:
                data.loc[:, group] = data[group].map(
                    lambda x: extract_parameter_value_as_int(x, extract_col_grouped_by))
        else:
            data[col_grouped_by] = data[col_grouped_by].map(
                lambda x: extract_parameter_value_as_int(x, extract_col_grouped_by))

    #data[new_col_related].astype(str)

    # Scatterplot create figure
    # _fig = plt.figure( figsize=(8*scale,40*scale))
    _fig = plt.figure(figsize=(10*width_multiplier, height*height_multiplier))

    # Create an axes instance
    ax1 = _fig.add_subplot(111)
    ax1.set_title(title,
                  fontsize=30  # *scale
                  )
    ax1.set_xlabel(x_title,
                   fontsize=30  # *scale
                   )
    ax1.set_ylabel(y_title,
                   rotation=90,
                   fontsize=30  # *scale
                   )
    # this sorts times and labels for display in the boxplot by the parameters of the boxplots
    # data_to_plot_arr, labels = zip(*sorted(zip(data_to_plot_arr,labels), key=lambda e: e[1] ))

    print("Plot prepared, preparing groups")

    groups = []
    # get the dataframes with their group names into list
    for group, group_df in data.groupby(col_grouped_by):
        # if the group is a tuple, trasform it to a nice string
        if isinstance(group, tuple):
            groups.append((", ".join([str(g) for g in group]), group_df))
        else:
            groups.append((group, group_df))

    # sort the list by the parameter so we can apply reasonable coloring
    print("Sorting groups")
    groups = sorted(groups, key=lambda x: x[0])
    current_size = max_marker_size
    # use seaborn to generate list of enough colors from a color pallete - it is graded
    colors = sns.color_palette(sns.dark_palette('cyan', n_colors=len(groups)), n_colors=len(groups))
    print("Adding scatter subplots")

    first = True
    for group, group_df in groups:
        if first:
            if isinstance(col_related, List):
                for c in col_related:
                    group_df = group_df.loc[group_df[c].map(comp_str_as_num).sort_values().index]
            else:
                group_df = group_df.loc[group_df[col_related].map(comp_str_as_num).sort_values().index]
            first = False
        # Create the scatterplot
        ax1.scatter(x=group_df[new_col_related].astype(str), y=group_df[col_examined], label=str(group), color=colors.pop(),
                    s=current_size)
        current_size -= (max_marker_size - min_marker_size) / len(groups)

    xlabels = [str(x[0]) for x in data.groupby(col_related)]

    # ax1.set_xticklabels(['1', '2', '5', '10', '50', '100', '500', '200'])
    print("Adding ticks, legend and labels")
    #print(xlabels)
    print(data[col_examined].min())
    print(data[col_examined].max())
    #ax1.set_xticklabels(xlabels)
    #ax1.set_xticks(xlabels)
    #ax1.set_yticks(y_ticks)
    ax1.tick_params(axis='x',
                    rotation=90,
                    labelsize=22  # *scale
                    )
    ax1.tick_params(axis='y',
                    labelsize=22  # *scale
                    )
    # ax1.grid(True)
    legend = plt.legend(loc="upper right",
                        bbox_to_anchor=(0, 0),
                        title=legend_title,
                        ncol=2,
                        prop={
                            'size': 20  # *scale
                        }
                        )
    legend.get_title().set_fontsize(22  # *scale
                                    )
    print("Saving figures to", graph_filename)
    _fig.savefig(graph_filename, bbox_inches="tight", dpi=dpi)
    plt.close(_fig)
    print("DONE")


def main():
    print("Starting graph creation, parsing arguments.")
    parser = setup_arguments()
    args = parser.parse_args()
    with open(args.config_graph, "r") as conf_file:
        conf = GraphScatterStepSchema(**json.load(conf_file))

    df, out_fp = prepare_data(args, conf)
    print("data prepared")
    if conf.separate_graphs_for_different_values_in_column is not None:
        raise NotImplementedError("This functionality is not ready.")
        gbc = df.groupby(conf.separate_graphs_for_different_values_in_column)
        counter = 0
        for group, group_df in gbc:
            legend_appendix = " Data are selected for value" + (
                "s " if not isinstance(group, str) else " ") + group + " in column" + ("s " if not isinstance(group,
                                                                                                              str) else " ") + conf.separate_graphs_for_different_values_in_column + "."

            out_split = out_fp.split(".")
            if not isinstance(group, str):
                group_out_fp = ".".join(out_split[:-1]) + "_" + \
                               str(counter) + "." + \
                               out_split[-1]
            else:
                group_out_fp = ".".join(out_split[:-1]) + "_" +\
                               str(counter) + "." +\
                               out_split[-1]
            counter += 1

            print_nice_scatterplot(data=group_df,
                                   graph_filename=group_out_fp,
                                   col_examined=conf.col_examined,
                                   col_grouped_by=conf.col_grouped_by,
                                   col_related=conf.col_related,
                                   title=conf.title,
                                   x_title=conf.x_title,
                                   y_title=conf.y_title,
                                   dpi=conf.dpi,
                                   height_multiplier=conf.height_multiplier,
                                   width_multiplier=conf.width_multiplier,
                                   legend_title=conf.legend_title + legend_appendix,
                                   max_y_val=conf.max_y_val,
                                   min_y_val=conf.min_y_val,
                                   convert_col_related_from_json=conf.convert_col_related_from_json,
                                   convert_col_grouped_by_from_json=conf.convert_col_grouped_by_from_json,
                                   extract_col_grouped_by=conf.extract_col_grouped_by,
                                   extract_col_related=conf.extract_col_related
                                   )

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
        print("Printing scatterplot to: ", out_fp, flush=True)
        print_nice_scatterplot(data=df,
                               graph_filename=out_fp,
                               col_examined=conf.col_examined,
                               col_grouped_by=conf.col_grouped_by,
                               col_related=conf.col_related,
                               title=conf.title,
                               x_title=conf.x_title,
                               y_title=conf.y_title,
                               dpi=conf.dpi,
                               height_multiplier=conf.height_multiplier,
                               width_multiplier=conf.width_multiplier,
                               legend_title=conf.legend_title,
                               max_y_val=conf.max_y_val,
                               min_y_val=conf.min_y_val,
                               convert_col_related_from_json=conf.convert_col_related_from_json,
                               convert_col_grouped_by_from_json=conf.convert_col_grouped_by_from_json,
                               extract_col_related=conf.extract_col_related,
                               extract_col_grouped_by=conf.extract_col_grouped_by)
