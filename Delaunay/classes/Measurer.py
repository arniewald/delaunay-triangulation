import numpy as np

from functions.measuring_functions import adjacency, mean_training_error, compute_edges_variance, compute_real_error

from classes.Classifier import Classifier

class Measurer:
    """
    Measures the different metrics of the training process.

    Attributes:
        - adjacency : adjacency relations between nodes of the Delaunay triangulation.
        - tri_error : mean training error of the points inside each triangle of the Delaunay triangulation.
        - metrics : lists with metrics of the training process.
    """
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
        """
        Computes the adjacency relations between nodes of the Delaunay triangulation.

        Args:
            - Classifier : classifier form which to compute the adjacency relations.

        Returns:
            - None
        """
        self.adjacency = adjacency(Classifier.tri,Classifier.out_hull)

    def measure_tri_error(self,Classifier: Classifier, e):
        """
        Computes the mean training error of the points inside each triangle of the Delaunay triangulation.

        Args:
            - Classifier : classifier form which to compute the mean training error inside the triangles.

        Returns: 
            - None
        """
        self.tri_error = mean_training_error(Classifier, e)

    def measure_training_error(self,Classifier: Classifier, e):
        """
        Computes the mean training error of the classifier.

        Args:
            - Classifier : classifier form which to compute the mean training error.
            - e : training error of each training point with respect to the classification estimation.

        Returns: 
            - None
        """
        average_error = sum(e)/len(Classifier.data)
        self.metrics['training_error'].append(average_error)

    def measure_error_variance(self,Classifier: Classifier, e):
        """
        Computes the training error variance of the classifier.

        Args:
            - Classifier : classifier form which to compute the training error variance.
            - e : training error of each training point with respect to the classification estimation.

        Returns: 
            - None
        """
        average_error = sum(e)/len(Classifier.data)
        self.metrics['error_variance'].append(np.sqrt(sum(e*e)/len(e) - average_error*average_error))

    def measure_edge_variance(self,Classifier: Classifier):
        """
        Computes the edge variance of the triangulation of the classifier.

        Args:
            - Classifier : classifier form which to compute the edge variance of its triangulation.

        Returns: 
            - None
        """
        self.metrics['edge_variance'].append(compute_edges_variance(Classifier))

    def measure_real_error(self,Classifier: Classifier, test_data, test_labels):
        """
        Computes the real error of the classifier.

        Args:
            - Classifier : classifier form which to compute the real error.
            - test_data : data from which real labels are known.
            - test_labels : real labels of test_data.

        Returns: 
            - None
        """
        self.metrics['real_error'].append(compute_real_error(Classifier, test_data, test_labels))
    


