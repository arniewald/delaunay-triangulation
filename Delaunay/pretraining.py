import os
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import json

from functions.initialization_functions import initialize_sample
from functions.reading_functions import extract_run_params, read_general_data
from functions.plotting_functions import plot_classifier

#Parameters
random.seed(0)
reset = True
data_name = 'anemia'
optimal_results_name = 'anemia'
path = str(os.getcwd()) + '\\jsons\\'+data_name+'_training_results.json'
optimal_results_path = str(os.getcwd()) + '\\jsons\\pretraining_'+optimal_results_name+'_results.json'

size_props = np.arange(0.05,0.5,0.05)
avws = np.arange(1,10.5,0.5)
samplings = ['random','entropic']

#If not existing, creates results json
if not os.path.exists(path):
    print('Creating destiny file...')
    f = open(path,'w')
    json.dump([],f)
    f.close()
    print('Destiny file created')
else:
    if reset:
        print("Reseting results for",data_name,"...")
        f = open(path,'w')
        json.dump([],f)
        f.close()
        print('Results reseted')



errors_dicts = []
len_validation_dict = {
    'circles': 1000,
    'beans': 347,
    'yeast': 52,
    'anemia': 10,
    'diabetes': 76
}

len_data_dict = {
    'circles': 10000,
    'beans': 3474,
    'yeast': 516,
    'anemia': 104,
    'diabetes': 758
}

data_params, data, labels, dim = read_general_data(data_name)
validation_indices = random.sample(range(len(data)),len_validation_dict[data_name])
data_indices = [int(i) for i in range(len(data)) if i not in validation_indices]
validation_data = data[validation_indices]
validation_labels = labels[validation_indices]
data = data[data_indices]
labels = labels[data_indices]
len_test = math.floor(len(data)*0.1)

#Pretraining
print('Ready to pretrain!')
for size_prop in size_props:
    for avw in avws:
        for sampling in samplings:
            from classes.Classifier import Classifier
            from classes.Trainer import Trainer
            from classes.Measurer import Measurer

            print('Size prop:',size_prop,'\tAvw:',avw,'\tSampling:',sampling)
            run_params = {
                "size_prop": size_prop,
                "sampling": sampling,
                "test_size": len_test, 
                "al": 0.035,               
                "errw": 0.5,               
                "avw": avw,
                "rep": 0,
                "bc_time": None,
                "mte_threshold": 0.5,
                "seed": 0
                }

            data_aux, labels_aux = data.copy(), labels.copy()
            

            Classifier = Classifier(data_aux, labels_aux, data_params, run_params)
            prelabels = Classifier.labels.copy()

            Trainer = Trainer(Classifier,run_params)
            postlabels = Classifier.labels[Classifier.sample,:].copy()
            
            Measurer = Measurer()
            Measurer.measure_training_error(Classifier,Trainer.e)
            Measurer.measure_real_error(Classifier,Classifier.test_data,Classifier.test_labels)

            run_params['training_error'] = Measurer.metrics['training_error'][-1]
            run_params['real_error'] = Measurer.metrics['real_error'][-1]
            print(run_params['real_error'])
            #errors_dicts.append(run_params)
            del Classifier
            del Trainer
            del Measurer

            with open(path,'r') as f:
                errors_dicts = json.load(f)
            f.close()
            errors_dicts.append(run_params)
            with open(path,'w') as f:
                json.dump(errors_dicts,f)
            f.close()

f = open(path)
results = json.load(f)
f.close()

#Find optimal parameters
results = [x for x in sorted(results, key=lambda item: item['real_error'])]
minimum_real_error = results[0]['real_error']
optimal_results = [x for x in results if x['real_error']==minimum_real_error]
#We should find a criteria for this
optimal = optimal_results[0]
training_error = optimal['training_error']
real_error = optimal['real_error']

#Classifying validation data
run_params = {
                "size_prop": optimal['size_prop'],
                "sampling": optimal['sampling'],
                "test_size": len_test, 
                "al": 0.035,               
                "errw": 0.5,               
                "avw": optimal['avw'],
                "rep": 0,
                "bc_time": None,
                "mte_threshold": 0.5,
                "seed": 0
                }
from classes.Classifier import Classifier
Classifier = Classifier(data, labels, data_params, run_params)
targets, errors, correct, incorrect = Classifier.classify(validation_data,validation_labels)
validation_error = len(incorrect)/len(targets)

#Writing results
pretraining_results = {
    'name': data_name,
    'training_params':
    {
        'length_training': len(data_indices),
        'length_testing': len_test,
        'length_validation': len(validation_indices)
    },
    'data_params': data_params,
    'pretraining_data': 
    {
        'validation_indices': validation_indices,
        'data_indices': data_indices
    },
    'optimal_parameters':
    {
        'size_prop': optimal['size_prop'],
        'avw': optimal['avw'],
        'sampling': optimal['sampling']
    },
    'training_error': training_error,
    'real_error': real_error,
    'validation_results':
    {
        'incorrect': incorrect,
        'validation_error': validation_error
    }
}

f = open(optimal_results_path,'w')
json.dump(pretraining_results,f)
f.close()

print('Pretraining completed')




