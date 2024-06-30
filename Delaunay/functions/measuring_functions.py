import numpy as np

from functions.classification_functions import classify

from classes.Classifier import Classifier

def adjacency(tri,out_hull):
    """
    Creates a dictionary indicating which points of the triangulation
    are adjacent to each other.

    Args:
        - tri: Delaunay triangulation.
        - out_hull: labels of the elements of sample not belonging to the convex hull.

    Returns:
        - adj: dictionary. Each key is an element of out_hull and its values are the labels of the points of the
               triangulation that are adjacent to the corresponding point. Note that the elements of the triangulation
               and the ones of out_hull are both indices of elements of sample.
    """
    adj = dict()
    print(len(tri.simplices))
    for i in out_hull:
        adj[i] = []
    for triangle in tri.simplices:
        try:
            for u in triangle:
                if u in out_hull:
                    for v in triangle:
                        adj[u].append(v)
            for key in adj.keys():
                adj[key] = list(set(adj[key]))
        except Exception as e:
            print('Exception at triangle',triangle,':',e) 
    return adj

def mean_training_error(Classifier: Classifier, e):
    """
    Computes the mean error of the training points inside each triangle of the Delaunay triangulation.

    Args:
        - data: the data to classify.
        - dim: dimension of data.
        - sample: labels of data belonging to the triangulation.
        - bc: barycentric coordinates of the points indexated by rem.
        - tri: Delaunay triangulation.
        - e: array with least square residuals.

    Returns:
        - tri_errors: dictionary with index of triangles as key and [barycenter of triangle, mean error of training
                      points inside the triangle] as value.
    """
    tri_errors = dict()
    for triangle in range(len(Classifier.tri.simplices)):
        #Mean error inside the triangle
        #errors = [e[i] for i in range(len(data)) if (bc[i][1]==triangle or i in sample[tri.simplices[triangle]])]
        errors = [e[i] for i in range(len(Classifier.rem)) if (Classifier.bc[i][1]==triangle)]
        if len(errors) != 0:
            error = sum(errors)/len(errors)
        else:
            error = 0
        #Barycenter
        barycenter = np.array([sum([Classifier.data[Classifier.sample[i]][j] for i in Classifier.tri.simplices[triangle]])/(Classifier.dim+1) for j in range(Classifier.dim)])
        tri_errors[triangle] = [barycenter,error]
    tri_errors = dict(sorted(tri_errors.items(), key=lambda item: item[1][1]))
    
    return tri_errors

def compute_real_error(Classifier: Classifier, points, real):
    """
    Computes the proportion of incorrectly predicted labels with respect to the real labels.

    Args:
        - points: points of which to compute the error.
        - dim: dimension of points.
        - tri: Delaunay triangulation.
        - trilabels: labels of the points of tri.
        - real: real labels of points.
    
    Returns:
        - error: if len(real)>0, the proportion of incorrectly predicted labels; else, 0.
    """
    error = 0
    targets, _, _,  incorrect = Classifier.classify(points, real)
    if len(targets)>0:
        error = len(incorrect)/len(targets)
    
    return error

def compute_edges_variance(Classifier: Classifier):
    """
    Computes the variance of the edges size of the triangulation.

    Args:
        - data: array with data.
        - dim: dimension of data.
        - sample: indices of data belonging to the triangulation.
        - tri: Delaunay triangulation.
        
    Returns:
        - sigma: variance of the edges size of the triangulation.
    """
    edges = []
    for triangle in Classifier.tri.simplices:
        for u in triangle:
            for v in triangle:
                if u!=v:
                    edges.append((u,v))
    edges = list(set(edges))
    sizes = np.array([np.sqrt(sum([(Classifier.data[Classifier.sample[edge[0]]][i]-Classifier.data[Classifier.sample[edge[1]]][i])**2 for i in range(Classifier.dim)])) for edge in edges])
    sigma = np.sqrt(sum(sizes*sizes)/len(sizes) -(sum(sizes)/len(sizes))**2)
    return sigma

