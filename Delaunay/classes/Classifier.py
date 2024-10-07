from functions.reading_functions import read_general_data
from functions.initialization_functions import sample_to_test, subtesselate, initialize_sample, reshape_labels
from functions.classification_functions import classify

class Classifier:
    """
    Data to be trained, parameters under which train it and new data to classify.

    Attributes:
        - dim_labels : nº of diferent possible labels - 1.
        - dim : dimension of data.
        - data : data from which the classifier will be created.
        - labels : labels of data.
        - sample : indices of the data that will be used as a classifier.
        - out_hull : indices of sample that do not belong to the convex hull boundary.
        - rem : indices of data that will be used for training.
        - data_params : parameters that characterize the data.
        - run_params : parameters that characterize how Classifier will be trained.
        - test_data : data used to compute real error.
        - test_labels : labels of test_data.
        - tri : Delaunay triangulation of points from data indexed by sample.
        - bc : barycentric coordinates of points from data indexed by rem with respect to tri.
        
    """
    def __init__(self, data, labels, data_params=None, run_params=None, sample=None, out_hull=None):
        """
        Initializes Classifier.

        Args:
            - data : data from which the classifier will be created
            - labels : labels of data
            - data_params : parameters that characterize the data
            - run_params : parameters that characterize how Classifier will be trained
            - sample : if not None, indices of the data that will be used as a classifier
            - out_hull : if not None, indices of sample that do not belong to the convex hull boundary.

        Returns:
            - None
        """
        if run_params!=None or data_params!=None:
            self.dim = len(data[0])
            self.dim_labels = len(set(labels))-1
            data, labels, test_data, test_labels = sample_to_test(data,labels,run_params)
            data, labels, sample, rem, out_hull = initialize_sample(data,labels,self.dim,run_params)
            labels = reshape_labels(data,labels,self.dim_labels)
            self.data = data
            self.labels = labels
            self.sample = sample
            self.out_hull = out_hull
            self.rem = rem
            self.data_params = data_params
            self.run_params = run_params
            self.test_data = test_data
            self.test_labels = test_labels

        else:
            self.dim = len(data[0])
            self.dim_labels = len(labels[0])
            self.data = data
            self.labels = labels
            self.sample = sample
            self.out_hull = out_hull
            self.rem = [int(i) for i in range(len(data)) if i not in sample]
            self.data_params = data_params
            self.run_params = run_params
            self.test_data = []
            self.test_labels = []

        tri, bc = subtesselate(data,sample,self.dim)
        self.tri = tri
        self.bc = bc

    def classify(self,test_data,test_labels):
        """
        Classifies data.

        Args:
            - test_data : data to be classified.
            - test_labels : real labels of test_data.

        Returns:
            Tuple containing
                - targets : estimated labels of test_data from the classification.
                - errors : absolute error of estimated labels.
                - correct : indices of elements of test_data correctly classified.
                - incorrect : indices of elements of test_data incorrectly classified.
        """
        if len(test_data)>0:
            targets, errors, correct, incorrect = classify(test_data, self.dim, self.tri, self.labels[self.sample,:], real=test_labels)
        else:
            targets, errors, correct, incorrect = classify(self.test_data, self.dim, self.tri, self.labels[self.sample,:], real=self.test_labels)
        return targets, errors, correct, incorrect

class TestClassifier(Classifier):
    """
    Classifier for one of the studied datasets. 

    Inherits:
        - Classifier
    """
    def __init__(self, data_name, run_params):
        data_params, data, labels, _ = read_general_data(data_name)
        super().__init__(data, labels, data_params, run_params)

