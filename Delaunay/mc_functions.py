import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.linalg import block_diag
from time import time

from functions import subtesselate, adjacency, compute_edges_variance

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

def mc_compute_eq_parameters(data, tri, sample, labels, dim_labels, bc, dim):
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
    B = labels[:,0]
    x0 = labels[sample,0]
    for i in range(1,dim_labels):
        B = np.concatenate([B,labels[:,i]])
        x0 = np.concatenate([x0,labels[sample,i]])
    
    M = np.zeros((len(data),len(sample)), dtype=np.float32)
    for i in range(len(data)):
        x = bc[i][1:(dim+3)]            #We extract index and barycentric coordinates of the i-th remaining point
        y = tri.simplices[int(x[0])]    #We extract the points of the triangulation containing the i-th remaining point
        M[i,y] = x[1:dim+2]
    
    A = block_diag(*([M]*dim_labels))

    return A, B, x0

def mc_classify(points, dim, tri, trilabels, real=None):
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

def mc_mean_training_error(data,dim,sample,bc,tri,e):
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
        errors = [e[i] for i in range(len(data)) if (bc[i][1]==triangle or i in sample[tri.simplices[triangle]])]
        if len(errors) != 0:
            error = sum(errors)/len(errors)
        else:
            error = 0
        #Barycenter
        barycenter = np.array([sum([data[sample[i]][j] for i in tri.simplices[triangle]])/(dim+1) for j in range(dim)])
        tri_errors[triangle] = [barycenter,error]
    return tri_errors

def mc_add_barycenters(data,labels,dim,sample,out_hull,tri,tri_errors,mte_threshold):
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
    added = []
    for triangle in tri_errors.keys():
        if tri_errors[triangle][1] > mte_threshold:
            data = np.concatenate([data,np.array([tri_errors[triangle][0]])])
            new_label = np.sum([labels[sample[i],:] for i in tri.simplices[triangle]], axis = 0)/(dim+1)
            labels = np.concatenate([labels,np.array([new_label])])
            sample = np.concatenate([sample,np.array([int(len(data)-1)])])
            out_hull = np.concatenate([out_hull,np.array([int(len(sample)-1)])])
            added.append(sample[-1])
    
    return data, labels, sample, out_hull, added

def mc_compute_real_error(points, dim, tri, trilabels, real):
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
        targets, _, _,  incorrect = mc_classify(points, dim, tri, trilabels,  real)
        error = len(incorrect)/len(targets)
    else:
        error = 0
    
    return error

def mc_delaunayization(data,sample,labels,dim,dim_labels):
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
    tri, bc = subtesselate(data,sample,dim)
    A, B, x0 = mc_compute_eq_parameters(data, tri, sample, labels, dim_labels, bc, dim)

    fun = lambda x: np.linalg.norm(A@x-B)**2
    bounds = [(0,1) for _ in range(len(sample)*dim_labels)]
    
    #We define the constraint matrix. We want: C*y = y1 + y2 <= np.ones(len(sample))
    C = np.zeros((len(sample),len(sample)*dim_labels), dtype = np.float32)
    for i in range(len(sample)):
        for j in range(dim_labels):
            C[i,i+j*len(sample)] = 1
    constraint = LinearConstraint(C,lb=np.zeros(len(sample)),ub=np.ones(len(sample)))

    start = time()
    print('Starting lsq fit...')
    y_aux = minimize(fun,x0,bounds=bounds,constraints=constraint).x
    end = time()
    print('Time to solve lsqfit: '+str(end-start))

    #This is extremly unoptimal but I don't know how to do it better right now: from matrix to column, from column to matrix.
    y = np.zeros((len(sample),dim_labels))
    for i in range(dim_labels):
        y[:,i] = y_aux[i*len(sample):(i+1)*len(sample)]
    e = abs(np.matmul(A,y_aux)-B)
    e_aux = np.zeros((len(data),dim_labels))
    for i in range(dim_labels):
        e_aux[:,i] = e[i*len(data):(i+1)*len(data)]
    e = np.array([np.sqrt(sum(e_aux[i]**2)) for i in range(len(e_aux))])
    return tri, bc, e, y

def mc_movepoints(data,labels,sample,out_hull,dim,dim_labels,it,al,bc_time=np.inf,mte_threshold=np.inf,filename = "",test_data=[],real=[]):
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
    avs, sigmas, maxs, evars, rerrs = [], [], [], [], []
    tri_errors = dict()
    added = []
    print("Iteration\t Mean error\t Error variance\t Maximum error")
    for i in range(it):
        print(i)
        try:
            tri, bc, e, labels[sample,:] = mc_delaunayization(data,sample,labels,dim,dim_labels)
            if (i+1)%bc_time==0:
                tri_errors = mc_mean_training_error(data,dim,sample,bc,tri,e)
                data, labels, sample, out_hull, added = mc_add_barycenters(data,labels,dim,sample,out_hull,tri,tri_errors,mte_threshold)
                print('Points added: ',len(added))
                out_hull = [int(x) for x in out_hull]
                tri, bc, e, labels[sample,:] = mc_delaunayization(data,sample,labels,dim,dim_labels)

            avs.append(sum(e)/len(data))
            sigmas.append(np.sqrt(sum(e*e)/len(e) - avs[i]*avs[i]))
            maxs.append(max(e))
            evars.append(compute_edges_variance(data,dim,sample,tri))
            rerrs.append(mc_compute_real_error(test_data, dim, tri, labels[sample], real))
            print(i,avs[i],sigmas[i],maxs[i],evars[i],rerrs[i])

            #We move the points
            start = time()
            L = len(out_hull)
            adj = adjacency(tri,out_hull)
            disin = np.zeros(L)
            for i in range(L):
                try:
                    disin[i] = sample[adj[out_hull[i]][np.argmax([sum((data[sample[j]]-data[sample[out_hull[i]]])**2) for j in adj[out_hull[i]]])]]
                except Exception as e:
                    print("Exception at node ",sample[out_hull[i]],": ",adj[out_hull[i]])
                    print("Exception: ",e)

            disin = [int(i) for i in disin]
            data[sample[out_hull]] += al*(data[disin]-data[sample[out_hull]])
            end = time()
            print('Time to move points: '+str(end-start))

        except Exception as e:
            print("Exception at time ",i,":",e)
            break

    tri, _, e, labels[sample,:] = mc_delaunayization(data,sample,labels,dim,dim_labels)
    print("Total final data: ",len(data))
    
    favs = open("errors/avs"+filename+".txt","w")
    fsigmas = open("errors/sigmas"+filename+".txt","w")
    fmaxs= open("errors/maxs"+filename+".txt","w")
    fevars = open("errors/evars"+filename+".txt","w")
    frerrs = open("errors/rerrs"+filename+".txt","w")
    for i in range(it-1):
        favs.write(str(avs[i])+"\t")
        fsigmas.write(str(sigmas[i])+"\t")
        fmaxs.write(str(maxs[i])+"\t")
        fevars.write(str(evars[i])+"\t")
        frerrs.write(str(rerrs[i])+"\t")
    favs.close()
    fsigmas.close()
    fmaxs.close()
    fevars.close()
    frerrs.close()

    return data, labels, sample, added, tri, e
