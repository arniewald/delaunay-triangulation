import numpy as np
import math
import random
from scipy.spatial import Delaunay, ConvexHull

def reshape_labels(data,labels,dim_labels):
    """
    Transforms the 1-D array of labels into an dim_labels-D array, where n = (nº of diferent labels - 1).

    Args:
        - data : the data to classify.
        - labels : original 1-D array of labels.
        - dim_labels : nº of diferent labels - 1.

    Returns:
        labels : the labels reshaped.
    """

    labels_aux = np.zeros((len(data),dim_labels+1))
    for i in range(len(data)):
            labels_aux[i,int(labels[i])] = 1
    labels = labels_aux[:,0:dim_labels]
    return labels

def refine(data,labels,dim,rep=0):
    """
    Adds points to the convex hull of the data.

    Args:
        - data : the data to refine.
        - labels : labels of the data.
        - dim : dimension of data.
        - rep : number of times to refine the convex hull.

    Returns:
        Tuple containing
            - new_data : data with the convex hull refined.
            - new_labels : labels of the previous data plus the ones from the new points added.
            - hull : indices of data belonging to the convex hull.
    """

    print("Refining convex hull...")
    hull = ConvexHull(data)
    hull = list(hull.vertices)
    new_data = data.copy()
    new_labels = labels.copy()
    for _ in range(rep):
        #We triangulate all the dataset
        tri = Delaunay(new_data)
        #We retrieve the faces in the convex hull
        faces = []
        for face in tri.simplices:
            face = [point for point in face if point in hull]
            faces.append(face)
        faces = [face for face in faces if len(face)==dim]
        data_hull = new_data[hull]
        center = sum(data_hull)/len(data_hull)
        minimum_distance = min([np.sqrt(sum((center-x)**2)) for x in data_hull])
        data_to_add = []
        labels_to_add = []
        L = len(new_data)
        for face in faces:
            point_to_add = sum([new_data[j] for j in face])/len(face)
            if np.sqrt(sum((center-point_to_add)**2))>=minimum_distance:
                data_to_add.append(point_to_add)
                label_to_add = sum([new_labels[j] for j in face])/len(face)
                labels_to_add.append(label_to_add)
                hull.append(L)
                L += 1
        new_data = np.concatenate([new_data,np.array(data_to_add)])
        new_labels = np.concatenate([new_labels,np.array(labels_to_add)])
    print('Convex hull refined ',rep,' times.')
    
    return new_data, new_labels, hull

def generate_shannon_entropies(data,labels,hull):
    """
    Generates the shannon entropy of each point of data.

    Args:
        - data : data from which to compute the shannon entropy.
        - labels : labels of data.
        - hull : indices of data belonging to the convex hull.

    Return:
        shannon_entropies : dictionary containing the shannon entropy of each point of data.
    """
    tri = Delaunay(data)
    out_hull = [i for i in range(len(data)) if i not in hull]
    adj = dict()
    for i in out_hull:
        adj[i] = []
    for triangle in tri.simplices:
        for u in triangle:
            if u in out_hull:
                for v in triangle:
                    adj[u].append(v)
    for key in adj.keys():
        adj[key] = list(set(adj[key]))    

    shannon_entropies = dict()
    for i in out_hull:
        shannon_entropy = 0
        n_neighbors = len(adj[i])
        label_values = set(labels[adj[i]])
        for j in label_values:
            n_labels = sum([1 for k in adj[i] if labels[k]==j])
            if n_labels!=0:
                shannon_entropy -= n_labels/n_neighbors*np.log(n_labels/n_neighbors)
        shannon_entropies[i] = shannon_entropy
    shannon_entropies = {k: v for k, v in sorted(shannon_entropies.items(), key=lambda item: item[1])}
    return shannon_entropies

def non_random_sample(shannon_entropies,size):
    """
    Draws a sample of the data, not including its convex hull. The probability is proportional to their shannon
    entropy.

    Args:
        - shannon_entropies : dictionary containing the shannon entropy of each point of data. 
        - size : size of the sample.

    Returns:
        out_hull : indices of data belonging to the drawn sample.
    """
    out_hull = []
    pop = list(shannon_entropies.keys())
    for _ in range(size):
        entropies_sum = sum([shannon_entropies[point] for point in pop])
        u = random.uniform(0,1)
        u *= entropies_sum
        total = 0
        for j in pop:
            total += shannon_entropies[j]
            if u<=total:
                out_hull.append(j)
                pop.remove(j)
                break
    return out_hull

def draw_initial_sample(data,labels,hull,size,sampling):
    """
    Draws the points of data that will be used to perform the Delaunay triangulation. 
    These points and their final estimated labels after training will consitute the classifier.

    Args:
        - data : data from which to draw the points.
        - labels : labels of data.
        - hull : boundary of the convex hull of data.
        - size : number of points to draw.
        - sampling : type of sampling to perform, either 'random' or 'entropic'.

    Returns:
        Tuple containing
            - sample : indices of the data that will be used as a classifier.
            - out_hull : indices of sample that do not belong to the convex hull boundary.
            - rem : indices of data that will be used for training.
    """
    length = len(hull)
    out_hull_size = max(0,size-length)

    if sampling=='random':
        out_hull = random.sample([i for i in range(len(data)) if i not in hull],out_hull_size)
    elif sampling=='entropic':
        shannon_entropies = generate_shannon_entropies(data,labels,hull)
        out_hull = non_random_sample(shannon_entropies,out_hull_size)
    
    sample = np.concatenate([hull,out_hull])
    sample = np.array([int(i) for i in sample])
    sample.sort()
    out_hull = [i for i in range(len(sample)) if sample[i] in out_hull]
    rem = np.array([int(i) for i in range(len(data)) if int(i) not in sample])
    return sample, out_hull, rem

def initialize_sample(data,labels,dim,run_params):
    """
    Refines the convex hull of data and the points to be sampled that will be used to perform the Delaunay triangulation

    Args:
        - data : array with initial data.
        - labels : labels of the data.
        - dim : dimension of data, number of features of data.
        - run_params : parameters that characterize how the data will be initialized and trained.
    
    Returns:
        Tuple containing
            - new_data : data with the convex hull refined.
            - new_labels : labels of the previous data plus the ones from the new points added.
            - sample : labels of the data selected for the sample.
            - rem : labels of the data not selected for the sample.
            - out_hull : labels of the elements of sample not belonging to the convex hull.
    """
    size_prop = run_params['size_prop']
    rep = run_params['rep']
    sampling = run_params['sampling']

    new_data, new_labels, hull = refine(data,labels,dim,rep)
    size = math.floor(size_prop*len(new_data))
    sample, out_hull, rem = draw_initial_sample(new_data,new_labels,hull,size,sampling)
    return new_data, new_labels, sample, rem, out_hull

def sample_to_test(data,labels,run_params):
    """
    Samples and retrieves a subset of data to perform tests.

    Args:
        - data : array with initial data from which to sample the test data.
        - labels : labels of the data.
        - run_params : parameters that characterize how the data will be initialized and trained.

    Returns:
        Tuple containing
            - rem_data : data not from the sample.
            - rem_labels : labels of data not from the sample.
            - test_data : data to be tested.
            - test_labels : labels of data to be tested.
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

def subtesselate(data,sample,dim):
    
    """ 
    Builds the Delaunay triangulation from a subsample of points containing the convex hull 
    and computes the barycentric coordinates from the other points.
    
    Args: 
        - data : original set of points.
        - sample : points from which to compute the Delaunay triangulation.
        - dim : dimension of the points of data.
    
    Returns:
        Tuple containing
            - tri : Delaunay triangulation. The indexation of the vertices comes from sample, not from data.
            - bc : barycentric coordinates of rem with respect to tri.
    """
    tri = Delaunay(data[sample])
    bc = []
    
    for i in range(len(data)):
        point = data[i]
        triangle = tri.find_simplex(point)
        b = tri.transform[triangle,:dim].dot(np.transpose(point - tri.transform[triangle,dim]))
        c = np.concatenate([b,[1-sum(b)]])
        bc.append(np.concatenate([[int(i),int(triangle)],c])) #Resulting array: index of point, index of triangle and barycentric coordinates
    return tri, bc

