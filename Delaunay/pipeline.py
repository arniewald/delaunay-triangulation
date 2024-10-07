import json
from functions.class_functions import fully_train
    

def simple_run(run_params,data_name,prefix,sufix,it,save,verbose):
    from classes.Classifier import TestClassifier
    from classes.Trainer import Trainer
    from classes.Measurer import Measurer
    from classes.Writer import Writer

    folder_name = data_name+'_'+prefix+'_'+sufix
    
    print(folder_name)
    TestClassifier = TestClassifier(data_name, run_params)
    Trainer = Trainer(TestClassifier,run_params)
    Measurer = Measurer(training_error=[],error_variance=[],edge_variance=[],real_error=[])
    fully_train(TestClassifier,Trainer,Measurer,it,save=save,verbose=verbose)
    Writer = Writer(data_name,TestClassifier.data_params,run_params,folder_name=folder_name)
    Writer.write_folder(TestClassifier, Measurer)

    del TestClassifier, Measurer, Trainer, Writer

f = open('jsons\\trainers_params.json')
trainers_params = json.load(f)
f.close()

data_names = ['yeast']
trainers_prefixes = ['balanced_gradient','error_gradient','distance_gradient']#['distance_gradient']#
trainers_sufixes = ['simple','barycenters','refine','full']#['barycenters']#

it = 100
save = False
verbose = False
exceptions = []
for data_name in data_names:
    for prefix in trainers_prefixes:
        for sufix in trainers_sufixes:
            run_params = trainers_params[prefix][sufix]
            run_params['size_prop'] = 0.1
            run_params['avw'] = 4.5
            run_params['sampling'] = 'random'
            try:
                simple_run(run_params,data_name,prefix,sufix,it,save,verbose)
            except Exception as e:
                exceptions.append((prefix,sufix,e))
