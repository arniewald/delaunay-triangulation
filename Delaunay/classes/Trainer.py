from classes.Classifier import Classifier

from functions.training_functions import delaunayization, train_step, add_barycenters

class Trainer:
    """
    Trains an specific classifier.

    Attributes:
        - al : measures the magnitude of the overall displacement.
        - errw :  weight of the error gradient.
        - avw :  weight of the averages equations.
        - bc_time : time at which to add barycenters.
        - mte_threshold : threshold above which the barycenters will be added.
        - e : training error of each training point with respect to the classification estimation.
        - err : estimated error of the points of the classifier.
    """
    def __init__(self,Classifier: Classifier, run_params):
        """
        Initializes Trainer.

        Args:
            - Classifier : Classifier to train.
            - run_params : parameters that characterize how the classifier will be trained.

        Returns:
            - None
        """
        self.al = run_params['al']
        self.errw = run_params['errw']
        self.avw = run_params['avw']
        self.bc_time = run_params['bc_time']
        self.mte_threshold = run_params['mte_threshold']
        
        e, err = delaunayization(Classifier,self.avw)
        self.e = e
        self.err = err

    def first_train(self,Classifier: Classifier):
        """
        Estimates the labels of the points of the triangulation.

        Args:
            - Classifier : Classifier to be trained.

        Returns:
            - None
        """
        self.e, self.err = delaunayization(Classifier,self.avw)
        

    def train(self,Classifier: Classifier):
        """
        Moves the points of the triangulation and estimates their labels.

        Args:
            - Classifier : Classifier to be trained.

        Returns:
            - None
        """
        train_step(Classifier,self.al,self.errw,self.err)
        self.e, self.err = delaunayization(Classifier,self.avw)
        
    def add_barycenters(self,Classifier: Classifier):
        """
        Adds barycenters of the triangulation to the classifying points.

        Args:
            - Classifier : Classifier to be trained.

        Returns:
            - added : indices of the data from classifier corresponding to the added barycenters.
        """
        added = add_barycenters(Classifier, self.e)
        return added
    