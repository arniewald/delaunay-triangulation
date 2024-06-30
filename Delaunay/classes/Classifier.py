from functions.reading_functions import read_general_data
from functions.initialization_functions import sample_to_test, subtesselate, initialize_sample, reshape_labels
from functions.classification_functions import classify

class Classifier:
    def __init__(self, data, labels, data_params=None, run_params=None, sample=None, out_hull=None):
        
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
        if len(test_data)>0:
            targets, errors, correct, incorrect = classify(test_data, self.dim, self.tri, self.labels[self.sample,:], real=test_labels)
        else:
            targets, errors, correct, incorrect = classify(self.test_data, self.dim, self.tri, self.labels[self.sample,:], real=self.test_labels)
        return targets, errors, correct, incorrect

class TestClassifier(Classifier):
    def __init__(self, data_name, run_params):
        data_params, data, labels, _ = read_general_data(data_name)
        super().__init__(data, labels, data_params, run_params)

class CirclesClassifier(TestClassifier):
    def __init__(self, run_params):
        super().__init__("circles",run_params)

class MoonsClassifier(TestClassifier):
    def __init__(self, run_params):
        super().__init__("moons",run_params)

class ClassificationClassifier(TestClassifier):
    def __init__(self, run_params):
        super().__init__("classification",run_params)

class IrisClassifier(TestClassifier):
    def __init__(self, run_params):
        super().__init__("iris",run_params)

class BeansClassifier(TestClassifier):
    def __init__(self, run_params):
        super().__init__("beans",run_params)

class YeastClassifier(TestClassifier):
    def __init__(self, run_params):
        super().__init__("yeast",run_params)

class FrancoClassifier(TestClassifier):
    def __init__(self, run_params):
        super().__init__("franco",run_params)

class SpheresClassifier(TestClassifier):
    def __init__(self, run_params):
        super().__init__("spheres",run_params)
