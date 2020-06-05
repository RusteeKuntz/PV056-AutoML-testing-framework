import argparse

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pv056_2019.utils import valid_path, convert_dict_to_parameter_pairs
import seaborn as sns;sns.set()

## agg backend is used to create plot as a .png file
mpl.use('agg')




def sort_boxplots_on_mean(e: [pd.Series, str]):
    return -e[0].mean()

def sort_boxplots_on_mean_inverse(e: [pd.Series, str]):
    return e[0].mean()

def sort_boxplots_on_label(e: [pd.Series, str]):
    return e[1]

def sort_boxplots_on_median(e: [pd.Series, str]):
    return e[0].median()

SORT_FUNCTIONS = {"label": sort_boxplots_on_label,
                  "mean": sort_boxplots_on_mean,
                  "inv_mean": sort_boxplots_on_mean_inverse,
                  "median": sort_boxplots_on_median}




def print_nice_scatterplot(data: pd.DataFrame,
                           graph_filename: str,
                           col_examined:str,
                           col_grouped_by:str,
                           col_related:str,
                           title:str,
                           legend_title:str,
                           x_title: str,
                           y_title: str,
                           max_y_val=None,
                           min_y_val=None,
                           convert_col_related_from_json=True):

    # preset configuration
    #scale=2
    max_marker_size = 1000#*scale
    min_marker_size = 1#*scale

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
    #_fig = plt.figure( figsize=(8*scale,40*scale))
    _fig = plt.figure(figsize=(8, 40))

    # Create an axes instance
    ax1 = _fig.add_subplot(111)
    ax1.set_title(title,
                 fontsize=25#*scale
                  )
    ax1.set_xlabel(x_title,
                   fontsize=25#*scale
                   )
    ax1.set_ylabel(y_title,
                   rotation=90,
                   fontsize=25#*scale
                   )
    # this sorts times and labels for display in the boxplot by the parameters of the boxplots
    #data_to_plot_arr, labels = zip(*sorted(zip(data_to_plot_arr,labels), key=lambda e: e[1] ))


    groups = []
    # get the dataframes with their group names into list
    for group, group_df in data.groupby(col_grouped_by):
        groups.append((group, group_df))

    # sort the list by the parameter so we can apply reasonable coloring
    groups = sorted(groups, key=lambda x: x[0])
    current_size = max_marker_size
    # use seaborn to generate list of enough colors from a color pallete - it is graded
    colors=sns.color_palette(sns.dark_palette('cyan', n_colors=len(groups)), n_colors=len(groups))
    for group, group_df in groups:
        # Create the scatterplot
        ax1.scatter(x=group_df[col_related], y=group_df[col_examined], label=str(group)+' % ', color=colors.pop(), s=current_size)
        current_size -= (max_marker_size-min_marker_size)/len(groups)

    #ax1.set_xticklabels(['1', '2', '5', '10', '50', '100', '500', '200'])
    ax1.set_yticks(y_ticks)
    ax1.tick_params(axis='x',
                    rotation = 90,
                    labelsize=22#*scale
                    )
    ax1.tick_params(axis='y',
                    labelsize=42#*scale
                    )
    #ax1.grid(True)
    legend = plt.legend(loc="upper right",
                        bbox_to_anchor=(0,0),
                        title=legend_title,
                        ncol=2,
                        prop={
                            'size': 16#*scale
                            }
                        )
    legend.get_title().set_fontsize(22#*scale
                                    )
    _fig.savefig(graph_filename, bbox_inches="tight", dpi=600)
    plt.close(_fig)




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
                   min_val=None,
                   max_val=None,
                   convert_col_related_from_json=True
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
    show_fliers = True
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
    if max_val is None:
        max_val = data[col_examined].max()
    if min_val is None:
        min_val = data[col_examined].min()
    tick = (max_val - min_val) / 40
    y_labels = np.concatenate([ np.arange(0, min_val-tick, -tick)[::-1], np.arange(0, max_val+6*tick, tick)])

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
    parser = argparse.ArgumentParser(description="PV056-AutoML-testing-framework")
    parser.add_argument(
        "-c",
        "--config-fs",
        type=lambda x: valid_path(x, "Invalid path to configuration json file!"),
        help="path to feature selection config file",
        required=True,
    )
    parser.add_argument(
        "-do",
        "--output-directory",
        help="path to a csv file which contains datasets used for FS and their respective result files",
        required=False,
    )
    parser.add_argument(
        "-di",
        "--datasets-csv-in",
        type=lambda x: valid_path(x, "Invalid path to input csv file!"),
        help="Path to csv file that contains previous data files mappings, locations and for example OD configurations",
        required=True,
    )
    args = parser.parse_args()

    print("Starting a graph generation step")
    # read dataset
    #df = pd.read_csv("results_dummy.csv", ",")
    import random
    num_of_inner_cols = 5
    df = pd.DataFrame([[j, *['{"argument": '+str(int(5*random.random()))+', "foo": "dammit"}' for i in range(num_of_inner_cols)], random.random()] for j in range(1000)], columns=["fold", *["step-"+str(i) for i in range(num_of_inner_cols)], "accuracy"])
    #df.to_csv("results_dummy.csv", index=False)


    od_method_name = 'TEST'

    print_boxplots(data=df,
                   graph_filename= od_method_name + '-boxplots-step-3',
                   col_examined="accuracy",
                   col_related="step-3",
                   #sort_func=,
                   title="Accuracy of classifiers\n for different n_estimators parameter\nof OD method IsolationForest\nsorted on the mean values\n",
                   x_title="parameter n_neighbors ",
                   y_title="Increase in accuracy after applying OD")
    #exit()

    print_nice_scatterplot(data=df,
                           graph_filename=od_method_name + "-scatterplot-step-2",
                           col_examined="accuracy",
                           col_grouped_by="step-0",
                           col_related="step-2",
                           title="Accuracy of classifiers\n for different n_neighbors parameter\nof OD method NearestNeighbors\nsorted on the mean values\n",
                           x_title="parameter n_neighbors",
                           y_title="change in gain after OD",
                           legend_title="% of removed outliers")

    exit()
    print_boxplots(data=df,
                   graph_filename= od_method_name +'-boxplots-removed',
                   col_examined="gain",
                   col_related = "removed",
                   sort_func=lambda e: -e[0].mean(),
                   title="Accuracy of classifiers\n for different n_estimators parameter\nof OD method IsolationForest\nsorted on the mean values\n",
                   x_title="% removed ",
                   y_title="Increase in accuracy after applying OD")


    sorted_df = df.sort_values(by="removed")
    print_nice_scatterplot(data=sorted_df,
                           graph_filename=od_method_name + '-scatterplot-removed',
                           col_examined="gain",
                           col_grouped_by="od_params",
                           col_related="removed",
                           title="Accuracy of classifiers\n for different n_neighbors parameter\nof OD method NearestNeighbors\nsorted on the mean values\n",
                           x_title="% of removed outliers",
                           y_title="change in gain after OD",
                           legend_title="parameter n_neighbors")



    # preset configuration
    grouped_bys = ["clf", "dataset"]

    # extract data
    for grouped_by in grouped_bys:
        gbc = df.groupby(grouped_by)
        title = 'of classifier' if grouped_by == "clf" else "for dataset"
        for group, group_df in gbc:
            print_nice_scatterplot(data=group_df,
                                   graph_filename=grouped_by+'_od_params/'+od_method_name+'-scatter-od_params-'+group+'.png',
                                   col_examined="gain",
                                   col_related="od_params",
                                   col_grouped_by="removed",
                                   x_title='n_neighbors value',
                                   y_title='Increase in gain after OD',
                                   title="Changes in accuracy "+title+" "+group+"\nbased on parameter n_neighbors \n of OD method "+od_method_name+" for different % of removed outliers\n",
                                   legend_title="% of removed outliers")

            print_boxplots(data=group_df,
                           graph_filename=grouped_by+'_od_params/'+od_method_name + '-boxplots-od_params-' + group,
                           col_examined="gain",
                           col_related="od_params",
                           sort_func=lambda e: -e[0].mean(),
                           title="Changes in accuracy "+title+" "+group+"\n for different n_neighbors parameter\nof OD method "+od_method_name+"\nsorted on the mean values\n",
                           x_title="n_neighbors value ",
                           y_title="Increase in accuracy after applying OD")


