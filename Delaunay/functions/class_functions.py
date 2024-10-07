from time import time

from classes.Classifier import Classifier
from classes.Trainer import Trainer
from classes.Measurer import Measurer

def add_barycenters_step(Classifier:Classifier,Trainer:Trainer,t,verbose=False):
    """
    Adds barycenters to the points of the triangulation.

    Args:
        - Classifier : Classifier the points of the triangulation of which barycenters are to be added.
        - Trainer : Trainer that adds the barycenters.
        - t : number of iteration of the full training.
        - verbose :  if True, prints the steps of the function.

    Returns:
        - added : indices of the data from classifier corresponding to the added barycenters. 
    """
    added = []
    if Trainer.bc_time!=None and (t+1)%Trainer.bc_time==0:
        start = time()
        if verbose:
                print('Adding barycenters...')
        added = Trainer.add_barycenters(Classifier)
        if verbose:
                print('Points added: ',len(added))
        Trainer.first_train(Classifier)
        end = time()
        if verbose:
                print('Time to add barycenters: '+str(end-start))
    return added

def premeasurement(Classifier: Classifier, Trainer: Trainer, Measurer: Measurer, test_data=[],test_labels=[], verbose=False):
    """
    Initializes the measurement of the metrics before starting the full training.

    Args:
        - Classifier : Classifier about to be trained.
        - Trainer : Trainer that will train the classifier.
        - Measurer : Measurer that performs and saves the metrics.
        - test_data : if not empty, data from which to compute the real error.
        - test_labels : if not empty, labels of test_data.
        - verbose : if True, prints the steps of the function.

    Returns:
        - None
    """
    Measurer.measure_training_error(Classifier, Trainer.e)
    if verbose:
        print('Training error measured')
    Measurer.measure_error_variance(Classifier, Trainer.e)
    if verbose:
        print('Error variance measured')
    Measurer.measure_edge_variance(Classifier)
    if verbose:
        print('Edge variance measured')
    Measurer.measure_real_error(Classifier,test_data,test_labels)
    if verbose:
        print('Real error measured')

def fully_train(Classifier:Classifier,Trainer:Trainer,Measurer:Measurer,it,test_data=[],test_labels=[],save=False,verbose=False):
    """
    Performs the full training of the classifier: moves the points of the triangulation, adds barycenters, estimates the labels and saves the metrics iteratively.

    Args:
        - Classifier : Classifier to be fully trained.
        - Trainer : Trainer that will perform the training, with the necessary parameters.
        - Measurer : Measurer that will measure and save the metrics of the training.
        - it : number of times to move points, estimate labels and compute metrics.
        - test_data : data from which to compute the real error.
        - test_labels : labels of test_data.
        - save : if True, saves the trajectories of the classification points, their estimated labels and their triangulations at each time.
        - verbose : if True, prints the steps of the function.

    Returns:
        Tuple containing
            - data: new data after moving it and adding barycenters.
            - labels: new labels.
            - sample: new indices of data from the triangulation.
            - added : indices of added barycenters.
            - long_data : trajectories of the classification points
            - long_labels : estimated labels of the classification points at each time.
            - long_tris : Delaunay triangulations at each time
    """
    
    added = []
    long_data, long_labels, long_tris = [], [], []
    if verbose:
        print("Iteration\tMean error\tError variance\tEdge length variance\tReal error")
    for i in range(it):
        #try:
            added = add_barycenters_step(Classifier,Trainer,i,verbose=verbose)
            if verbose:
                print('Barycenters added:',len(added))
            Trainer.train(Classifier)
            if verbose:
                print('Data trained')
            Measurer.measure_training_error(Classifier, Trainer.e)
            if verbose:
                print('Training error measured')
            Measurer.measure_error_variance(Classifier, Trainer.e)
            if verbose:
                print('Error variance measured')
            Measurer.measure_edge_variance(Classifier)
            if verbose:
                print('Edge variance measured')
            Measurer.measure_real_error(Classifier,test_data,test_labels)
            if verbose:
                print('Real error measured')
            if verbose:
                print(i,Measurer.metrics['training_error'][i],Measurer.metrics['error_variance'][i],Measurer.metrics['edge_variance'][i],Measurer.metrics['real_error'][i])
            if save:
                long_data.append(Classifier.data[Classifier.sample])
                long_labels.append(Classifier.labels[Classifier.sample,:])
                long_tris.append(Classifier.tri)
            """ except Exception as e:
                print("Exception at time ",i,":",e)
                break """

    if verbose:
        print("Total final data: ",len(Classifier.data))
    
    return added, long_data, long_labels, long_tris
