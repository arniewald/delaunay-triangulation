import numpy as np
import random
from scipy.spatial import ConvexHull

def sample_to_test(data,labels,run_params):
    """
    Samples and retrieves a subset of data to perform tests.

    Args:
        - data: array with initial data from which to sample the test data.
        - labels: labels of the data.
        - size: size of the sample.

    Returns:
        - rem_data: data not from the sample.
        - rem_labels: labels of data not from the sample.
        - test_data: data to be tested.
        - test_labels: labels of data to be tested.
    """
    test_size = run_params['test_size']
    hull = ConvexHull(data)
    hull = list(hull.vertices)
    indices = random.sample([int(i) for i in range(len(data)) if i not in hull],test_size)
    test_data = data[indices].copy()
    test_labels = labels[indices].copy()
    rem_data = data[[i for i in range(len(data)) if i not in indices]].copy()
    rem_labels = labels[[i for i in range(len(data)) if i not in indices]].copy()
    return rem_data, rem_labels, test_data, test_labels


def classify(points, dim, tri, trilabels, real=None):
    """
    Classifies points in multiple labels given a Delaunay triangulation.
    It computes the weighted value of each possible label given the barycentric coordinates of the triangle's vertices of 
    each point, then takes the one with maximum value.

    Args:
        - points: points to be classified. They must lie within the convex hull of the triangulation.
        - dim: dimension of data.
        - tri: Delaunay triangulation.
        - trilabels: labels of the points of tri.
        - real: real values of the labels of points (in case we want to compare the estimated labels with them)

    Returns:
        - targets: estimated labels of points.
        - errors: if real!=None, errors of the estimated labels.
        - correct: if real!=None, indices of points whose estimated label is the same as the real one.
        - incorrect: if real!=None, indices of points whose estimated label is not the same as the real one.
    """
    bc = []
    errors, correct, incorrect = [], [], []
    for i in range(len(points)):
        point = points[i]
        triangle = tri.find_simplex(point)
        b = tri.transform[triangle,:dim].dot(np.transpose(point - tri.transform[triangle,dim]))
        c = np.concatenate([b,[1-sum(b)]])
        bc.append(np.concatenate([[int(i),int(triangle)],c]))

    A = np.concatenate([trilabels,np.array([np.ones(trilabels.shape[0])-np.sum(trilabels,axis=1)]).transpose()],axis=1)
    targets = [np.argmax(np.sum(np.array(bc[i][2:])*A[tri.simplices[int(bc[i][1])]].transpose(), axis=1)) for i in range(len(points))]

    if real.any()!=None:
        errors = np.abs(targets-real)
        correct = [i for i in range(len(targets)) if errors[i]<=0]
        incorrect = [i for i in range(len(targets)) if errors[i]>0]
    return targets, errors, correct, incorrect
