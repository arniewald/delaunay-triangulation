from time import time

from classes.Classifier import Classifier
from classes.Trainer import Trainer
from classes.Measurer import Measurer

def add_barycenters_step(Classifier:Classifier,Trainer:Trainer,t):
    added = []
    if Trainer.bc_time!=None and (t+1)%Trainer.bc_time==0:
        start = time()
        print('Adding barycenters...')
        added = Trainer.add_barycenters(Classifier)
        print('Points added: ',len(added))
        Trainer.first_train(Classifier)
        end = time()
        print('Time to add barycenters: '+str(end-start))
    return added

def fully_train(Classifier:Classifier,Trainer:Trainer,Measurer:Measurer,it,test_data=[],test_labels=[],save=False,verbose=False):
    """
    Performs the estimation of labels and movement of points as many times as indicated.
    Also writes, for each iteration: the sum of estimated errors, the sum of the squared residuals, the variance of the estimated
    error and the real error (only if function f applied).

    Args:
        - data: array with data.
        - labels: labels of the data.
        - sample: labels of data belonging to the triangulation.
        - out_hull: labels of the elements of sample not belonging to the convex hull.
        - dim: dimension of data.
        - dim_labels: nº of diferent labels - 1
        - it: number of times to move the points.
        - al: measures the magnitude of the overall displacement.
        - bc_time: time at which to add barycenters.
        - mte_threshold: threshold above which the barycenters will be added.
        - filename: core name of the file where to write the errors.
        - test_data: if len(test_data)>0, data with which to compute the real error.
        - real: if len(real)>0, real labels of data with which to compute the real error.

    Returns:
        - data: new data after moving it and adding barycenters.
        - labels: new labels.
        - sample: new indices of data from the triangulation.
        - added: indices of added data.
        - tri: final Delaunay triangulation.
        - e: error of points from triangulation.
    """
    
    added = []
    long_data, long_labels, long_tris = [], [], []
    if verbose:
        print("Iteration\tMean error\tError variance\tEdge length variance\tReal error")
    for i in range(it):
        print(i)
        print(len(Classifier.rem))
        try:
            added = add_barycenters_step(Classifier,Trainer,i)
            print('Barycenters added:',len(added))
            Trainer.train(Classifier)
            print('Data trained')
            Measurer.measure_training_error(Classifier, Trainer.e)
            print('Training error measured')
            Measurer.measure_error_variance(Classifier, Trainer.e)
            print('Error variance measured')
            Measurer.measure_edge_variance(Classifier)
            print('Edge variance measured')
            Measurer.measure_real_error(Classifier,test_data,test_labels)
            print('Real error measured')
            if verbose:
                print(i,Measurer.metrics['training_error'][i],Measurer.metrics['error_variance'][i],Measurer.metrics['edge_variance'][i],Measurer.metrics['real_error'][i])
            if save:
                long_data.append(Classifier.data[Classifier.sample])
                long_labels.append(Classifier.labels[Classifier.sample,:])
                long_tris.append(Classifier.tri)
        except Exception as e:
            print("Exception at time ",i,":",e)
            break

    if verbose:
        print("Total final data: ",len(Classifier.data))
    
    return added, long_data, long_labels, long_tris
