import os
import pandas as pd

from functions.plotting_functions import plot_data

def save_results(data_name,filename,data,labels,sample,rem,added,test_data,test_labels,correct,incorrect,tri,dim,dim_labels,err_dict):
    path = str(os.getcwd())+'\\results\\' + data_name
    errors_path = path + '\\errors'
    data_path = path + '\\data'
    media_path = path + '\\media'
    filename = filename
    if os.path.isdir(path)==False:
        os.makedirs(path)
        os.makedirs(errors_path)
        os.makedirs(data_path)
        os.makedirs(media_path)

    data_plot = plot_data(data,labels,dim_labels,sample,rem,added,test_data,test_labels,correct,incorrect,tri,dim)
    err_csv = pd.DataFrame.from_dict(err_dict)
    err_csv.to_csv(errors_path+'\\'+filename+'.csv')
    data_plot.savefig(media_path+'\\'+filename+'.png')

def generate_filename(data_name, run_params):
    filename = data_name
    for param in run_params.keys():
        if run_params[param]!=None:
            if isinstance(run_params[param],str):
                filename = filename + '_' + param + '_' + run_params[param]
            else:
                filename = filename + '_' + param + str(round(run_params[param],3))
    return filename