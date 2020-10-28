# import argparse
#
# import pandas as pd
# import matplotlib as mpl
#
# from pv056_2019.utils import valid_path
#
# from pv056_2019.visualize.main_box import print_boxplots
#
# from pv056_2019.visualize.main_scatter import print_nice_scatterplot
# import seaborn as sns;sns.set()
#
# ## agg backend is used to create plot as a .png file
# mpl.use('agg')
#
#
# def main():
#     parser = argparse.ArgumentParser(description="PV056-AutoML-testing-framework")
#     parser.add_argument(
#         "-c",
#         "--config-fs",
#         type=lambda x: valid_path(x, "Invalid path to configuration json file!"),
#         help="path to feature selection config file",
#         required=True,
#     )
#     parser.add_argument(
#         "-do",
#         "--output-directory",
#         help="path to a csv file which contains datasets used for FS and their respective result files",
#         required=False,
#     )
#     parser.add_argument(
#         "-di",
#         "--datasets-csv-in",
#         type=lambda x: valid_path(x, "Invalid path to input csv file!"),
#         help="Path to csv file that contains previous data files mappings, locations and for example OD configurations",
#         required=True,
#     )
#     args = parser.parse_args()
#
#
#
#     # read dataset
#     df = pd.read_csv("results_dummy.csv", ",")
#     #import random
#     #num_of_inner_cols = 5
#     #df = pd.DataFrame([[j, *['{"argument": '+str(int(5*random.random()))+', "foo": "dammit"}' for i in range(num_of_inner_cols)], random.random()] for j in range(1000)], columns=["fold", *["step-"+str(i) for i in range(num_of_inner_cols)], "accuracy"])
#     #df.to_csv("results_dummy.csv", index=False)
#
#
#     od_method_name = 'TEST'
#
#     print_boxplots(data=df,
#                    graph_filename= od_method_name + '-boxplots-step-3',
#                    col_examined="accuracy",
#                    col_related="step-3",
#                    #sort_func=,
#                    title="Accuracy of classifiers\n for different n_estimators parameter\nof OD method IsolationForest\nsorted on the mean values\n",
#                    x_title="parameter n_neighbors ",
#                    y_title="Increase in accuracy after applying OD")
#     #exit()
#
#     print_nice_scatterplot(data=df,
#                            graph_filename=od_method_name + "-scatterplot-step-2",
#                            col_examined="accuracy",
#                            col_grouped_by="step-0",
#                            col_related="step-2",
#                            title="Accuracy of classifiers\n for different n_neighbors parameter\nof OD method NearestNeighbors\nsorted on the mean values\n",
#                            x_title="parameter n_neighbors",
#                            y_title="change in gain after OD",
#                            legend_title="% of removed outliers")
#
#     exit()
#     print_boxplots(data=df,
#                    graph_filename= od_method_name +'-boxplots-removed',
#                    col_examined="gain",
#                    col_related = "removed",
#                    sort_func=lambda e: -e[0].mean(),
#                    title="Accuracy of classifiers\n for different n_estimators parameter\nof OD method IsolationForest\nsorted on the mean values\n",
#                    x_title="% removed ",
#                    y_title="Increase in accuracy after applying OD")
#
#
#     sorted_df = df.sort_values(by="removed")
#     print_nice_scatterplot(data=sorted_df,
#                            graph_filename=od_method_name + '-scatterplot-removed',
#                            col_examined="gain",
#                            col_grouped_by="od_params",
#                            col_related="removed",
#                            title="Accuracy of classifiers\n for different n_neighbors parameter\nof OD method NearestNeighbors\nsorted on the mean values\n",
#                            x_title="% of removed outliers",
#                            y_title="change in gain after OD",
#                            legend_title="parameter n_neighbors")
#
#
#
#     # preset configuration
#     grouped_bys = ["clf", "dataset"]
#
#     # extract data
#     for grouped_by in grouped_bys:
#         gbc = df.groupby(grouped_by)
#         title = 'of classifier' if grouped_by == "clf" else "for dataset"
#         for group, group_df in gbc:
#             print_nice_scatterplot(data=group_df,
#                                    graph_filename=grouped_by+'_od_params/'+od_method_name+'-scatter-od_params-'+group+'.png',
#                                    col_examined="gain",
#                                    col_related="od_params",
#                                    col_grouped_by="removed",
#                                    x_title='n_neighbors value',
#                                    y_title='Increase in gain after OD',
#                                    title="Changes in accuracy "+title+" "+group+"\nbased on parameter n_neighbors \n of OD method "+od_method_name+" for different % of removed outliers\n",
#                                    legend_title="% of removed outliers")
#
#             print_boxplots(data=group_df,
#                            graph_filename=grouped_by+'_od_params/'+od_method_name + '-boxplots-od_params-' + group,
#                            col_examined="gain",
#                            col_related="od_params",
#                            sort_func=lambda e: -e[0].mean(),
#                            title="Changes in accuracy "+title+" "+group+"\n for different n_neighbors parameter\nof OD method "+od_method_name+"\nsorted on the mean values\n",
#                            x_title="n_neighbors value ",
#                            y_title="Increase in accuracy after applying OD")
#
#
