import os
import json
import numpy as np
import pandas as pd

class Reader():
    def __init__(self,folder_name=None,to_match={}):
        self.path = str(os.getcwd()) + '\\classifiers\\'

        if folder_name!=None:
            dir_list = os.listdir(self.path)
            dir_list.remove('general_metadata.json')
            if folder_name in dir_list:
                self.folder_path = self.path + folder_name + '\\'
            else:
                print('Error: no folder named ',folder_name)
                self.folder_path = self.path
        else:
            print('No folder name provided; matching other arguments...')
            general_metadata_path = str(os.getcwd()) + '\\classifiers\\general_metadata.json'

            f = open(general_metadata_path)
            general_metadata = json.load(f)
            f.close()

            best_match = None
            folder_names = list(general_metadata.keys())
            max_score = 0
            args = list(to_match.keys())
            for folder_name in folder_names:
                score = 0
                for arg in args:
                    if arg in general_metadata[folder_name].keys():
                        if general_metadata[folder_name][arg] == to_match[arg]:
                            score+=1
                if score>=max_score:
                    best_match=folder_name
                    max_score=score

            if max_score==0:
                print('Error: no matches found')
                self.folder_path = self.path
            else:
                self.folder_path = self.path + best_match +'\\'

    def read_classifier(self):
        from classes.Classifier import Classifier
        df = pd.read_csv(self.folder_path+'classifier.csv')
        metadata_path = self.folder_path + 'metadata.json'
        f = open(metadata_path)
        metadata = json.load(f)
        f.close()
        names_data = metadata['names_data']
        names_labels = metadata['names_labels']

        data = []
        for name_data in names_data:
            data.append(df[name_data].tolist())
        data = np.array(data).T

        labels = []
        for name_label in names_labels:
            labels.append(df[name_label].tolist())
        labels = np.array(labels).T

        sample = [i for i in range(len(df)) if df.iloc[i]['sample']==True]
        out_hull = [i for i in range(len(df)) if df.iloc[i]['out_hull']==True]
        
        Classifier = Classifier(data, labels, sample=sample, out_hull=out_hull)
        return Classifier

    def read_measurer(self):
        from classes.Measurer import Measurer
        df = pd.read_csv(self.folder_path+'metrics.csv')
        training_error = df.training_error.tolist()
        error_variance = df.error_variance.tolist()
        edge_variance = df.edge_variance.tolist()
        real_error = df.real_error.tolist()
        Measurer = Measurer(training_error,error_variance,edge_variance,real_error)
        return Measurer
    



