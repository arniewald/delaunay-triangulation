import os
import numpy as np
import random
import json

from functions.initialization_functions import initialize_sample
from functions.reading_functions import extract_run_params, read_general_data


data_name = 'diabetes'
folder_name = 'jsons/'
path = folder_name + data_name + '_training_results.json'

f = open(path)
results = json.load(f)
f.close()

keys = ['size_prop','avw','sampling','real_error']
results = [{'size_prop': x['size_prop'], 'avw': x['avw'], 'sampling': x['sampling'],'training_error': x['training_error'],'real_error': x['real_error']} for x in sorted(results, key=lambda item: item['real_error'])]
minimum_real_error = results[0]['real_error']
optimal_results = [x for x in results if x['real_error']==minimum_real_error]

#print(len(results)-len(optimal_results))
for x in optimal_results:#set(optimal_results):
    print(x)
    #print('size_prop=',x['size_prop'],'avw=',x['avw'])