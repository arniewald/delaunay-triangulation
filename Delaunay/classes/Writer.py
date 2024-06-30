import os
import shutil
import pandas as pd
import json
from datetime import datetime

from classes.Classifier import Classifier
from classes.Trainer import Trainer
from classes.Measurer import Measurer

class Writer:
    def __init__(self,data_name,data_params,run_params,names_data=None,names_labels=None,folder_name=None):
        if folder_name!=None:
            self.folder_name = folder_name
        else:
            self.folder_name = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        cwd = str(os.getcwd())
        self.folder_path = cwd+'\\classifiers\\'+self.folder_name+'\\'
        if os.path.isdir(self.folder_path)==False:
            os.mkdir(self.folder_path)

        self.data_name = data_name
        self.data_params = data_params
        self.run_params = run_params
        self.names_data = names_data
        self.names_labels = names_labels

    
    def write_classifier(self, Classifier: Classifier):
        d = dict()
        if self.names_data==None:
            self.names_data = ['x'+str(i) for i in range(Classifier.dim)]
        for i in range(Classifier.dim):
            name_data = self.names_data[i]
            d[name_data] = list(Classifier.data[:,i])

        if self.names_labels==None:
            self.names_labels = ['y'+str(i) for i in range(Classifier.dim_labels)]
        for i in range(Classifier.dim_labels):
            name_label = self.names_labels[i]
            d[name_label] = list(Classifier.labels[:,i])
        
        d['sample']=[(i in Classifier.sample) for i in range(len(Classifier.data))]
        d['out_hull']=[(i in Classifier.sample and list(Classifier.sample).index(i) in Classifier.sample) for i in range(len(Classifier.data))]

        df = pd.DataFrame.from_dict(d)
        print(self.folder_name)
        df.to_csv(self.folder_path+'classifier.csv')

    def write_metadata(self):
        d = dict()
        d['data_name'] = self.data_name
        d['names_data'] = self.names_data
        d['names_labels'] = self.names_labels
        for key in self.data_params.keys():
            d[key] = self.data_params[key]
        for key in self.run_params.keys():
            d[key] = self.run_params[key]
        with open(self.folder_path+'metadata.json','w') as f:
            json.dump(d,f)
    
    def write_metrics(self, Measurer: Measurer, metrics_names = None):
        metrics = dict()
        if metrics_names == None:
            metrics_names = Measurer.metrics.keys()
        for metric in metrics_names:
            metrics[metric] = Measurer.metrics[metric]
        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(self.folder_path+'metrics.csv')

    def add_general_metadata(self):
        general_metadata_path = str(os.getcwd()) + '\\classifiers\\general_metadata.json'

        f = open(general_metadata_path)
        general_metadata = json.load(f)
        f.close()

        general_metadata[self.folder_name] = dict()
        general_metadata[self.folder_name]['data_name'] = self.data_name
        for key in self.data_params.keys():
             general_metadata[self.folder_name][key] = self.data_params[key]
        for key in self.run_params.keys():
             general_metadata[self.folder_name][key] = self.run_params[key]

        #Inneficient: it rewrites data all the time
        with open(general_metadata_path,'w') as f:
            json.dump(general_metadata,f)

    def update_general_metadata(self):
        path = str(os.getcwd()) + '\\classifiers\\'
        general_metadata_path = str(os.getcwd()) + '\\classifiers\\general_metadata.json'

        f = open(general_metadata_path)
        general_metadata = json.load(f)
        f.close()
        dir_list = os.listdir(path)
        dir_list.remove('general_metadata.json')

        changes = False
        removed = []
        folder_names = list(general_metadata.keys())
        for folder_name in folder_names:
            if folder_name not in dir_list:
                removed.append(folder_name)
                del general_metadata[folder_name]
                changes = True
        
        with open(general_metadata_path,'w') as f:
            json.dump(general_metadata,f)

        if changes:
            print('Folders removed: ',removed)
        else:
            print('No updates')

    def write_folder(self, Classifier: Classifier, Measurer: Measurer, metrics_names):
        self.write_classifier(Classifier)
        self.write_metadata()
        self.write_metrics(Measurer=Measurer,metrics_names=metrics_names)
        self.add_general_metadata()
        

        
        


        


        
