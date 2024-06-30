import numpy as np

from functions.measuring_functions import adjacency, mean_training_error, compute_edges_variance, compute_real_error

from classes.Classifier import Classifier

class Measurer:
    def __init__(self,training_error=[],error_variance=[],edge_variance=[],real_error=[]):
        self.adjacency = dict()
        self.tri_error = dict()
        self.metrics = {
            'training_error': training_error,
            'error_variance': error_variance,
            'edge_variance': edge_variance,
            'real_error': real_error
        }
    
    def measure_adjacency(self,Classifier: Classifier):
        self.adjacency = adjacency(Classifier.tri,Classifier.out_hull)

    def measure_tri_error(self,Classifier: Classifier, e):
        self.tri_error = mean_training_error(Classifier, e)

    def measure_training_error(self,Classifier: Classifier, e):
        average_error = sum(e)/len(Classifier.data)
        self.metrics['training_error'].append(average_error)

    def measure_error_variance(self,Classifier: Classifier, e):
        average_error = sum(e)/len(Classifier.data)
        self.metrics['error_variance'].append(np.sqrt(sum(e*e)/len(e) - average_error*average_error))

    def measure_edge_variance(self,Classifier: Classifier):
        self.metrics['edge_variance'].append(compute_edges_variance(Classifier))

    def measure_real_error(self,Classifier: Classifier, test_data, test_labels):
        self.metrics['real_error'].append(compute_real_error(Classifier, test_data, test_labels))
    


