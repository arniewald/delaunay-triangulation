import os
import numpy as np
import math
import random
import pandas as pd
from scipy.spatial import Delaunay, ConvexHull
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from time import time
from qpsolvers import solve_qp

from plotting_functions import *

def reshape_labels(data,labels):
    """
    Transforms the 1-D array of labels into an dim_labels-D array, where n = (nº of diferent labels - 1).

    Args:
        - data: the data to classify
        - labels: original 1-D array of labels

    Returns:
        - labels: the labels reshaped
        - dim_labels: nº of diferent labels - 1
    """

    dim_labels = len(set(labels))
    labels_aux = np.zeros((len(data),dim_labels))
    for i in range(len(data)):
            labels_aux[i,int(labels[i])] = 1
    dim_labels -= 1
    labels = labels_aux[:,0:dim_labels]
    return labels, dim_labels

def refine(data,labels,dim,rep=0):
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
    Refines the convex hull of data and selects sample to perform Delaunay triangulation.

    Args:
        - data: array with initial data.
        - labels: labels of the data.
        - size: size of the desired sample. If convex hull is bigger, the sample will just contain the convex hull.
        - dim: dimension of data: each element of data has dim features and one label.
        - rep: number of times to refine the convex hull.
    
    Returns:
        - new data: data reordered (first elements belong to sample) and with refinement of the complex hull.
        - sample: labels of the data selected for the sample.
        - rem: labels of the data not selected for the sample.
        - out_hull: labels of the elements of sample not belonging to the convex hull.
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

def adjacency(tri, out_hull):
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

def subtesselate(data,sample,dim):
    
    """ 
    Builds the Delaunay triangulation from a subsample of points containing the convex hull 
    and computes the barycentric coordinates from the other points.
    
    Args: 
        - data: original set of points.
        - sample: points from which to compute the Delaunay triangulation.
        - dim: dimension of the points of data.
    
    Returns:
        - tri: Delaunay triangulation. The indexation of the vertices comes from sample, not from data.
        - bc: barycentric coordinates of rem with respect to tri.
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

def compute_eq_parameters(data, tri, sample, labels, dim_labels, bc, dim, avw):
    """
    Computes the matrix and the column from which to estimate the labels of the points of the triangulation.

    Args:
        - data: the data to classify.
        - tri: Delaunay triangulation.
        - dim_labels: nº of diferent labels - 1
        - sample: labels of data belonging to the triangulation.
        - bc: barycentric coordinates of the points indexated by rem.
        - dim: dimension of data.
    
    Returns:
        - A, B: the matrices that describe the least squares problem: ||Ax-B||**2=0.
        - x0: the seed of the algorith to minimize the lsq problem, taken as the original labels.
    """

    #dim_labels = nº of possible labels - 1
    #B = labels[:,0]
    rem = [i for i in range(len(data)) if i not in sample]
    B = labels[rem,0]
    for i in range(1,dim_labels):
        #B = np.concatenate([B,labels[:,i]])
        B = np.concatenate([B,labels[rem,i]])
    
    
    print('\tBuilding barycentric coordinates equations...')
    M = np.zeros((len(rem),len(sample)), dtype=np.float32)
    for i in range(len(rem)):
        x = bc[rem[i]][1:(dim+3)]            #We extract index and barycentric coordinates of the i-th remaining point
        y = tri.simplices[int(x[0])]    #We extract the points of the triangulation containing the i-th remaining point
        M[i,y] = x[1:dim+2]
    A = block_diag(*([M]*dim_labels))
    print('\tBarycentric coordinates equations built')

    print('\tBuilding averages equations...')
    adj = adjacency(tri,range(len(sample)))
    print('\t \t Adjacency built')
    S = np.zeros((len(sample),len(sample)), dtype=np.float32)
    for i in range(len(sample)):
        nodes = [node for node in adj[i] if node != i]
        len_nodes = len(nodes)
        S[i,i] = -len_nodes
        for node in nodes:
            S[i,node] = 1
    A = np.concatenate([A,block_diag(*([S]*dim_labels))])
    B = np.concatenate([B,np.zeros(len(sample)*dim_labels)])
    print('\tAverages equations built')

    print('\tBuilding weighting matrix...')
    W = np.zeros(((len(rem)+len(sample))*dim_labels,(len(rem)+len(sample))*dim_labels))
    for i in range(len(rem)*dim_labels):
        W[i,i] = 1
    for i in range(len(rem)*dim_labels, (len(rem)+len(sample))*dim_labels):
        W[i,i] = avw
    print('\tWeighting matrix built')

    return A, B, M, S, W

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
    else:
        return targets

def mean_training_error(data,dim,sample,rem,bc,tri,e):
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
    for triangle in range(len(tri.simplices)):
        #Mean error inside the triangle
        #errors = [e[i] for i in range(len(data)) if (bc[i][1]==triangle or i in sample[tri.simplices[triangle]])]
        errors = [e[i] for i in range(len(rem)) if (bc[i][1]==triangle)]
        if len(errors) != 0:
            error = sum(errors)/len(errors)
        else:
            error = 0
        #Barycenter
        barycenter = np.array([sum([data[sample[i]][j] for i in tri.simplices[triangle]])/(dim+1) for j in range(dim)])
        tri_errors[triangle] = [barycenter,error]
    tri_errors = dict(sorted(tri_errors.items(), key=lambda item: item[1][1]))
    
    return tri_errors

def add_barycenters(data,labels,dim,sample,out_hull,tri,tri_errors,run_params):
    """
    Adds the barycenter of those triangles whose mean training error is greater than a threshold to the triangulation.

    Args:
        - data: the data to classify.
        - labels: labels of the data.
        - dim: dimension of data.
        - sample: indices of data belonging to the triangulation.
        - out_hull: indices of the elements of sample not belonging to the convex hull.
        - tri: Delaunay triangulation.
        - tri_errors: dictionary containing the barycenter the mean training error of each triangle.
        - mte_threshold: threshold above which the barycenters will be added.

    Returns:
        - data: new data with the added barycenters.
        - labels: new labels with the ones of the added barycenters.
        - sample: new indices of data belonging to the triangulation with the ones of the added barycenters.
        - out_hull: new indices of the elements of sample not belonging to the convex hull with the ones of the added barycenters.
        - added: indices of the added barycenters.
    """
    #For the time being, so that we do not add too many points, we add the top 10
    mte_threshold = run_params['mte_threshold']
    d = dict()
    for i in range(len(tri_errors)-20,len(tri_errors)):
        key = list(tri_errors.keys())[i]
        d[key] = tri_errors[key]
    tri_errors = d
    added = []
    for triangle in tri_errors.keys():
        #if tri_errors[triangle][1] > mte_threshold:
            data = np.concatenate([data,np.array([tri_errors[triangle][0]])])
            new_label = np.sum([labels[sample[i],:] for i in tri.simplices[triangle]], axis = 0)/(dim+1)
            labels = np.concatenate([labels,np.array([new_label])])
            sample = np.concatenate([sample,np.array([int(len(data)-1)])])
            out_hull = np.concatenate([out_hull,np.array([int(len(sample)-1)])])
            added.append(sample[-1])
    
    return data, labels, sample, out_hull, added

def compute_real_error(points, dim, tri, trilabels, real):
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
    if len(real)>0:
        targets, _, _,  incorrect = classify(points, dim, tri, trilabels,  real)
        error = len(incorrect)/len(targets)
    else:
        error = 0
    
    return error

def compute_edges_variance(data,dim,sample,tri):
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
    for triangle in tri.simplices:
        for u in triangle:
            for v in triangle:
                if u!=v:
                    edges.append((u,v))
    edges = list(set(edges))
    sizes = np.array([np.sqrt(sum([(data[sample[edge[0]]][i]-data[sample[edge[1]]][i])**2 for i in range(dim)])) for edge in edges])
    sigma = np.sqrt(sum(sizes*sizes)/len(sizes) -(sum(sizes)/len(sizes))**2)
    return sigma

def delaunayization(data,sample,labels,dim,dim_labels,run_params):
    """
    Performs Delaunay triangulation, computes barycentric coordinates, estimates labels and estimates error with multiple labels.

    Args:
        - data: array with data.
        - sample: labels of data belonging to the triangulation.
        - labels: labels of the data.
        - dim: dimension of data.
        - dim_labels: nº of diferent labels - 1
    
    Returns:
        - tri: Delaunay triangulation.
        - bc: barycentric coordinates of the points indexated by rem.
        - e: errors of each estimated label with every possible label except the last one.
        - y: estimated values for each label.
    """
    avw = run_params['avw']
    tri, bc = subtesselate(data,sample,dim)
    start = time()
    A, B, M, S, W = compute_eq_parameters(data, tri, sample, labels, dim_labels, bc, dim, avw)
    
    #To minimize: ||Ax-B||^2
    #Minimize 0.5*xT*P*x+qT*x s. t. Gx<=h, lb<=x<=ub
    #P = 2*AT*A, q=-2*AT*B
    At = np.transpose(A)
    P = csc_matrix(2*np.matmul(np.matmul(At,W),A))
    q = -2*np.matmul(np.matmul(At,W),B)
    
    #We define the constraint matrix. We want: C*y = y1 + y2 <= np.ones(len(sample))
    G = np.zeros((len(sample),len(sample)*dim_labels), dtype = np.float32)
    for i in range(len(sample)):
        for j in range(dim_labels):
            G[i,i+j*len(sample)] = 1
    h = np.ones(len(sample))
    G = csc_matrix(G)
    #lb = 0, ub = 1
    lb = np.zeros(len(sample)*dim_labels)
    ub = np.ones(len(sample)*dim_labels)
    end = time()
    print('Time to build parameters: '+str(end-start))

    start = time()
    print('Starting lsq fit...')
    y_aux = solve_qp(P,q,G=G,h=h,A=None,b=None,lb=lb,ub=ub,solver='piqp')
    end = time()
    print('Time to solve lsqfit: '+str(end-start))

    #This is extremly unoptimal but I don't know how to do it better right now: from matrix to column, from column to matrix.
    len_rem = len(data)-len(sample)
    y = np.zeros((len(sample),dim_labels))
    for i in range(dim_labels):
        y[:,i] = y_aux[i*len(sample):(i+1)*len(sample)]
    e = abs(np.matmul(A[:len_rem*dim_labels,:],y_aux)-B[:len_rem*dim_labels])
    
    
    e_aux = np.zeros((len_rem,dim_labels))
    for i in range(dim_labels):
        e_aux[:,i] = e[i*len_rem:(i+1)*len_rem]
    e = np.array([np.sqrt(sum(e_aux[i]**2)) for i in range(len(e_aux))])
    
    start = time()
    Mt = np.transpose(M)
    P2 = csc_matrix(2*np.matmul(Mt,M))
    q2 = -2*np.matmul(e,M)
    """ e_bc = np.zeros((len_rem,dim_labels))
    for i in range(dim_labels):
        e_bc[:,i] = e[i*len_rem:(i+1)*len_rem]
    e_bc = np.array([np.sqrt(sum(e_bc[i]**2)) for i in range(len(e_bc))])
    e_av = np.zeros((len(sample),dim_labels))
    for i in range(dim_labels):
        e_av[:,i] = e[len_rem*dim_labels+i*len(sample):len_rem*dim_labels+(i+1)*len(sample)]
    e_av = np.array([np.sqrt(sum(e_av[i]**2)) for i in range(len(e_av))])
    e = np.concatenate([e_bc,e_av])
    start = time()
    A2 = np.concatenate([M,S])
    A2t = np.transpose(A2)
    W2 = np.zeros((int(W.shape[0]/dim_labels),int(W.shape[0]/dim_labels)))
    W2[0:M.shape[0],0:M.shape[0]] = (1-avw)*np.identity(M.shape[0])
    W2[M.shape[0]:,M.shape[0]:] = avw*np.identity(S.shape[0])
    print(len(sample),e.shape,A.shape,A2.shape,W2.shape)
    #Mt = np.transpose(M)
    P2 = csc_matrix(2*np.matmul(np.matmul(A2t,W2),A2))
    q2 = -2*np.matmul(e,np.matmul(W2,A2)) """
    lb = np.zeros(len(sample))
    ub = np.ones(len(sample))
    err = solve_qp(P2,q2,G=None,h=None,A=None,b=None,lb=lb,ub=ub,solver='piqp')
    print('Time to solve quadratic: '+str(end-start))

    return tri, bc, e, y, err

def movepoints_step(data,sample,tri,out_hull,err,run_params):
    al = run_params['al']
    errw = run_params['errw']
    start = time()
    L = len(out_hull)
    adj = adjacency(tri,out_hull)
    disin = np.zeros(L)
    errin = np.zeros(L)
    for i in range(L):
        try:
            errin[i] = sample[adj[out_hull[i]][np.argmax([(err[j]-err[out_hull[i]]) for j in adj[out_hull[i]]])]]
            disin[i] = sample[adj[out_hull[i]][np.argmax([sum((data[sample[j]]-data[sample[out_hull[i]]])**2) for j in adj[out_hull[i]]])]]
        except Exception as e:
            print("Exception at node ",sample[i],": ",adj[i])
            print("Exception: ",e)
    errin = [int(i) for i in errin]
    disin = [int(i) for i in disin]
    data[sample[out_hull]] += al*(errw*(data[errin]-data[out_hull])+(1-errw)*(data[disin]-data[out_hull]))
    end = time()
    print('Time to move points: '+str(end-start))
    return data

def add_barycenters_step(data,labels,dim,dim_labels,sample,rem,out_hull,tri,bc,e,tri_errors,err,run_params,t):
    added = []
    bc_time = run_params['bc_time']
    if bc_time!=None and (t+1)%bc_time==0:
        start = time()
        print('Adding barycenters...')
        tri_errors = mean_training_error(data,dim,sample,rem,bc,tri,e)
        print('Errors calculated')
        data, labels, sample, out_hull, added = add_barycenters(data,labels,dim,sample,out_hull,tri,tri_errors,run_params)
        print('Points added: ',len(added))
        out_hull = [int(x) for x in out_hull]
        tri, bc, e, labels[sample,:], err = delaunayization(data,sample,labels,dim,dim_labels,run_params)
        end = time()
        print('Time to add barycenters: '+str(end-start))
    return data, labels, sample, out_hull, added, tri, bc, e, labels[sample,:], err

def movepoints(data,labels,sample,rem,out_hull,dim,dim_labels,run_params,it,filename = "",test_data=[],real=[],save=False,verbose=False):
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
    
    err_dict = dict()
    err_dict['avs'], err_dict['sigmas'], err_dict['evars'], err_dict['rerrs'] = [], [], [], []
    tri_errors = dict()
    added = []
    long_data, long_labels, long_tris = [], [], []
    if verbose:
        print("Iteration\tMean error\tError variance\tEdge length variance\tReal error")
    for i in range(it):
        print(i)
        print(len(rem))
        try:
            tri, bc, e, labels[sample,:], err = delaunayization(data,sample,labels,dim,dim_labels,run_params)
            data, labels, sample, out_hull, added, tri, bc, e, labels[sample,:], err = add_barycenters_step(data,labels,dim,dim_labels,sample,rem,out_hull,tri,bc,e,tri_errors,err,run_params,i)
            err_dict['avs'].append(sum(e)/len(data))
            err_dict['sigmas'].append(np.sqrt(sum(e*e)/len(e) - err_dict['avs'][i]*err_dict['avs'][i]))
            err_dict['evars'].append(compute_edges_variance(data,dim,sample,tri))
            err_dict['rerrs'].append(compute_real_error(test_data, dim, tri, labels[sample], real))
            if verbose:
                print(i,err_dict['avs'][i],err_dict['sigmas'][i],err_dict['evars'][i],err_dict['rerrs'][i])
            if save:
                long_data.append(data[sample])
                long_labels.append(labels[sample,:])
                long_tris.append(tri)
            #We move the points
            data = movepoints_step(data,sample,tri,out_hull,err,run_params)
        except Exception as e:
            print("Exception at time ",i,":",e)
            break

    tri, _, e, labels[sample,:], err = delaunayization(data,sample,labels,dim,dim_labels,run_params)
    if verbose:
        print("Total final data: ",len(data))
    
    

    return data, labels, sample, added, tri, e, err_dict, long_data, long_labels, long_tris

def save_results(data_name,filename,data,labels,sample,rem,added,test_data,test_labels,correct,incorrect,tri,dim,dim_labels,err_dict):
    path = str(os.getcwd())+'\\results\\' + data_name
    errors_path = path + '\\errors'
    data_path = path + '\\data'
    media_path = path + '\\media'
    filename = filename
    if os.path.isdir(path)==False:
        os.makedirs(path)
        os.makedirs(errors_path)
        os.makedirs(data_path)
        os.makedirs(media_path)

    data_plot = plot_data(data,labels,dim_labels,sample,rem,added,test_data,test_labels,correct,incorrect,tri,dim)
    err_csv = pd.DataFrame.from_dict(err_dict)
    err_csv.to_csv(errors_path+'\\'+filename+'.csv')
    data_plot.savefig(media_path+'\\'+filename+'.png')

def generate_filename(data_name, run_params):
    filename = data_name
    for param in run_params.keys():
        if run_params[param]!=None:
            if isinstance(run_params[param],str):
                filename = filename + '_' + param + '_' + run_params[param]
            else:
                filename = filename + '_' + param + str(round(run_params[param],3))
    return filename