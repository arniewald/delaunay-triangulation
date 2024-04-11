import numpy as np
import math
from scipy.sparse import dok_array
from scipy.spatial import Delaunay, ConvexHull
from scipy.optimize import lsq_linear, nnls, minimize, LinearConstraint
from scipy.linalg import block_diag
import random
from time import time

def f(x,y):
    """
    Example function to give values to the points.
    """
    return np.exp(-x*x)*np.sin(y)

def refine(data,labels,dim,rep=1):
    """
    Deprecated. Adds points to the convex hull.

    Args:
        - data: data to which add the new points.
        - labels: labels of the data.
        - dim: dimension of data
        - rep: number of refinements to perform, each adds the middle point 
               of each edge connecting two consecutive points of the convex hull.
    
    Returns:
        - old_data: data before the refinement.
        - new_data: data after the refinement.
        - in_hull: indices of the data points belonging to the refined hull-
        - L: length of in_hull.
    """
    #THIS NEEDS TO BE ADAPTED TO >2 DIMENSIONS!!!
    hull = ConvexHull(data)
    hull = list(hull.vertices)
    nohull = [i for i in range(len(data)) if i not in hull]
    new_labels = np.array(np.concatenate([labels[hull],labels[nohull]]))
    in_hull_data = [list(data[i]) for i in hull]
    L = len(in_hull_data)
    out_hull_data = np.array([data[i] for i in range(len(data)) if i not in hull])
    for i in range(rep):
        for j in range(L):
            point = []
            k = (2*j+1)%(2*L-1)
            for l in range(dim):
                point.append((in_hull_data[2*j][l] + in_hull_data[k][l])/dim)
            #For the time being, we assign the average label to the refinement points:
            point.append((in_hull_data[j][dim]+in_hull_data[j][dim])/2)
            in_hull_data.insert(2*j+1,np.array(point))
        L = len(in_hull_data)
    
    in_hull = [int(i) for i in range(L)]
    in_hull_data = np.array(in_hull_data)
    new_data = np.array(np.concatenate([in_hull_data,out_hull_data]))
    
    return new_data, new_labels, in_hull, L

def initialize_sample(data,labels,size,dim,rep=1):
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
    new_data, new_labels, in_hull, L = refine(data,labels,dim,rep)
    length = len(new_data)
    sample = random.sample([int(i) for i in range(L,length)],max(0,size-L))
    sample = np.concatenate([in_hull,sample])
    sample.sort()
    sample = np.array([int(i) for i in sample])
    rem = np.array([int(i) for i in range(length) if int(i) not in sample])
    if size>L:
        out_hull = list(range(L,size))
    else:
        out_hull = []
    return new_data, new_labels, sample, rem, out_hull

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
    for i in out_hull:
        adj[i] = []
    for triangle in tri.simplices:
        for u in triangle:
            if u in out_hull:
                for v in triangle:
                    adj[u].append(v)
    for key in adj.keys():
       adj[key] = list(set(adj[key]))    
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


def compute_eq_parameters(data, tri, sample, labels, bc, dim):
    """
    Computes the matrix and the column from which to estimate the labels of the points of the triangulation.

    Args:
        - data: array with data.
        - tri: Delaunay triangulation.
        - sample: indices of data belonging to the triangulation.
        - labels: labels of the data.
        - bc: barycentric coordinates of the points indexated by rem.
        - dim: dimension of data.
    
    Returns:
        - A, B: the matrices that describe the least squares problem: ||Ax-B||**2=0.
    """
    B = labels
    A = dok_array((len(data),len(sample)), dtype=np.float32)
    for i in range(len(data)):
        x = bc[i][1:(dim+3)]            #We extract index and barycentric coordinates of the i-th remaining point
        y = tri.simplices[int(x[0])]    #We extract the points of the triangulation containing the i-th remaining point
        A[i,y] = x[1:dim+2]
    return A, B



def compute_eq_parameters_with_rank(data, tri, rem, sample, labels, bc, dim):
    """
    Computes the matrix and the column from which to estimate the labels of the points of the triangulation.
    It only computes them so that the equation is not undetermined.

    Args:
        - data: array with data.
        - tri: Delaunay triangulation.
        - rem: indices of data not belonging to the triangulation.
        - sample: indices of data belonging to the triangulation.
        - labels: labels of the data.
        - bc: barycentric coordinates of the points indexated by rem.
        - dim: dimension of data.
    
    Returns:
        - A, B: the matrices that describe the least squares problem: ||Ax-B||**2=0.
    """
    #Check what points of triangulation do not have equations
    appear = []
    for i in range(len(rem)):
        appear = appear + list(tri.simplices[int(bc[rem[i]][1])])
    appear = set(appear)
    to_add = list(set(range(len(sample))).difference(appear))
    print(to_add)

    len_rem = len(rem)
    Aaux = np.zeros((len(data),len(sample)), dtype=np.float32)
    for i in range(len_rem):
        x = bc[rem[i]][1:(dim+3)]            #We extract index and barycentric coordinates of the i-th remaining point
        y = tri.simplices[int(x[0])]    #We extract the points of the triangulation containing the i-th remaining point
        Aaux[i,y] = x[1:dim+2]
    for i in range(len(to_add)):
        Aaux[i+len_rem,to_add[i]] = 1
    B = list(labels[rem])+list(labels[sample[to_add]])
    L = len(rem)+len(to_add)
    appear = list(appear)
    while (np.linalg.matrix_rank(Aaux)<len(sample)) and L<len(data):
        i = appear.pop()
        Aaux[L,i] = 1
        B.append(labels[i])
        L += 1
    if np.linalg.matrix_rank(Aaux)>=len(sample):
        print("Rank achieved")
    A = dok_array((L,len(sample)), dtype = np.float32)
    for i in range(L):
        A[i] = Aaux[i]
    B = np.array(B)
    return A, B

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

def compute_real_error(points, dim, tri, trilabels, threshold, real):
    """
    Computes the proportion of incorrectly predicted labels with respect to the real labels.

    Args:
        - points: points of which to compute the error.
        - dim: dimension of points.
        - tri: Delaunay triangulation.
        - trilabels: labels of the points of tri.
        - threshold: threshold at which a point is classified to class 1.
        - real: real labels of points.
    
    Returns:
        - error: if len(real)>0, the proportion of incorrectly predicted labels; else, 0.
    """
    if len(real)>0:
        targets, _, _,  incorrect = classify(points, dim, tri, trilabels, threshold, real)
        error =  len(incorrect)/len(targets)
    
    else:
        error = 0
    
    return error

def mean_training_error(data,dim,sample,bc,tri,e):
    """
    Computes the mean error of the training points inside each triangle of the Delaunay triangulation.

    Args:
        - data: the data to classify.
        - dim: dimension of data.
        - sample: indices of data belonging to the triangulation.
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

def add_barycenters(data,labels,dim,sample,out_hull,tri,tri_errors,mte_threshold):
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
            #I do not know how to assign a label to a new point; for the time being, average
            new_label = sum([labels[sample[i]] for i in tri.simplices[triangle]])/(dim+1)
            labels = np.concatenate([labels,np.array([new_label])])
            sample = np.concatenate([sample,np.array([int(len(data)-1)])])
            out_hull = np.concatenate([out_hull,np.array([int(len(sample)-1)])])
            added.append(sample[-1])
    
    return data, labels, sample, out_hull, added

def movepoints_step(data, sample, out_hull, tri, err, al, errw=0.5):
    """
    Moves one time the points of sample not in the convex hull according to the error and the distance gradient.

    Args:
        - data: array with data.
        - sample: indices of data belonging to the triangulation.
        - out_hull: labels of the elements of sample not belonging to the convex hull.
        - tri: Delaunay triangulation.
        - err: estimated errors of the points of the triangulation.
        - al: measures the magnitude of the overall displacement.
        - errw: weight given to the error gradient (weight given to the distance gradient is 1-errw).
    
    Returns:
        - None
    """
    start = time()
    L = len(out_hull)
    adj = adjacency(tri,out_hull)
    errin = np.zeros(L)
    disin = np.zeros(L)
    for i in range(L):
        try:
            errin[i] = sample[adj[out_hull[i]][np.argmax([(err[j]-err[out_hull[i]]) for j in adj[out_hull[i]]])]]
            disin[i] = sample[adj[out_hull[i]][np.argmax([sum((data[sample[j]]-data[sample[out_hull[i]]])**2) for j in adj[out_hull[i]]])]]
        except Exception as e:
            print("Exception at node ",sample[out_hull[i]],": ",adj[out_hull[i]])
            print("Exception: ",e)

    errin = [int(i) for i in errin]
    disin = [int(i) for i in disin]
    data[sample[out_hull]] += al*(errw*(data[errin]-data[sample[out_hull]])+(1-errw)*(data[disin]-data[sample[out_hull]]))
    end = time()
    print('Time to move points: '+str(end-start))
    return None



def delaunayization(data,sample,labels,dim,lb=-np.inf,ub=np.inf):
    """
    Performs Delaunay triangulation, computes barycentric coordinates, estimates labels and estimates error.

    Args:
        - data: array with data.
        - sample: indices of data belonging to the triangulation.
        - labels: labels of the data.
        - dim: dimension of data.
        - lb: lower boundary of estimated labels.
        - ub: upper boundary of estimated labels.
    
    Returns:
        - tri: Delaunay triangulation.
        - bc: barycentric coordinates of the points indexated by rem.
        - e: array with least square residuals.
        - err: array with estimated errors.
        - y: estimated values for each label.
    """
    tri, bc = subtesselate(data,sample,dim)
    A, B = compute_eq_parameters(data, tri, sample, labels, bc, dim)

    start = time()
    y = lsq_linear(A,B,bounds=(lb,ub),lsq_solver='lsmr')['x']
    end = time()
    print('Time to solve lsqfit: '+str(end-start))

    e = abs(np.matmul(A.todense(),y)-B)
    start = time()
    try:
        err = nnls(A.todense(),e)[0] #Faster than lsq_linear
    except RuntimeError as ex:
        print(ex)
        print("Trying lsq_linear...")
        err = lsq_linear(A,e,bounds=(0,np.inf),lsq_solver='lsmr')['x']
    end = time()
    print('Time to solve quadratic: '+str(end-start))

    return tri, bc, e, err, y



def movepoints(data,labels,sample,out_hull,dim,it,al,errw=0.5,lb=-np.inf,ub=np.inf,threshold=0.5,bc_time=np.inf,mte_threshold=np.inf,filename = "",test_data=[],real=[]):
    """
    Performs the estimation of labels and movement of points as many times as indicated.
    Also writes, for each iteration: the sum of estimated errors, the sum of the squared residuals, the variance of the estimated
    error and the real error (only if function f applied).

    Args:
        - data: array with data.
        - labels: labels of the data.
        - sample: indices of data belonging to the triangulation.
        - rem: indices of data not belonging to the triangulation.
        - out_hull: labels of the elements of sample not belonging to the convex hull.
        - dim: dimension of data.
        - it: number of times to move the points.
        - al: measures the magnitude of the overall displacement.
        - errw: weight given to the error gradient (weight given to the distance gradient is 1-errw).
        - lb: lower boundary of estimated labels.
        - ub: upper boundary of estimated labels.
        - threshold: value from which map binary labels.
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
        - err: estimated error of points from triangulation.
    """
    avs, sigmas, maxs, evars, rerrs = [], [], [], [], []
    tri_errors = dict()
    added = []
    print("Iteration\t Mean error\t Error variance\t Maximum error")
    for i in range(it):
        try:
            tri, bc, e, err, labels[sample] = delaunayization(data,sample,labels,dim,lb,ub)
            if (i+1)%bc_time==0:
                tri_errors = mean_training_error(data,dim,sample,bc,tri,e)
                data, labels, sample, out_hull, added = add_barycenters(data,labels,dim,sample,out_hull,tri,tri_errors,mte_threshold)
                print('Points added: ',len(added))
                tri, bc, e, err, labels[sample] = delaunayization(data,sample,labels,dim,lb,ub)
            avs.append(sum(e)/len(data))
            sigmas.append(np.sqrt(sum(e*e)/len(e) - avs[i]*avs[i]))
            maxs.append(max(e))
            evars.append(compute_edges_variance(data,dim,sample,tri))
            rerrs.append(compute_real_error(test_data, dim, tri, labels[sample], threshold, real))
            print(i,avs[i],sigmas[i],maxs[i],evars[i],rerrs[i])

            movepoints_step(data, sample, out_hull, tri, err, al, errw)

        except Exception as e:
            print("Exception at time ",i,":",e)
            break

    tri, bc, e, err, labels[sample] = delaunayization(data,sample,labels,dim,lb,ub)
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

    return data, labels, sample, added, tri, e, err

def sample_to_test(data,labels,size):
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
    hull = ConvexHull(data)
    hull = list(hull.vertices)
    indices = random.sample([int(i) for i in range(len(data)) if i not in hull],size)
    test_data = data[indices].copy()
    test_labels = labels[indices].copy()
    rem_data = data[[i for i in range(len(data)) if i not in indices]].copy()
    rem_labels = labels[[i for i in range(len(data)) if i not in indices]].copy()
    return rem_data, rem_labels, test_data, test_labels

def classify(points, dim, tri, trilabels, threshold=0.5, real=None):
    """
    Classifies points in two labels given a Delaunay triangulation.
    It computes the weighted value of each possible label given the barycentric coordinates of the triangle's vertices of 
    each point, then takes the one with maximum value.

    Args:
        - points: points to be classified. They must lie within the convex hull of the triangulation.
        - dim: dimension of data.
        - tri: Delaunay triangulation.
        - trilabels: labels of the points of tri.
        - threshold: threshold at which a point is classified to class 1.
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
    
    A = np.zeros((len(points),len(trilabels)),np.float32)
    for i in range(len(points)):
        x = bc[i][1:(dim+3)]            #We extract index and barycentric coordinates of the i-th remaining point
        y = tri.simplices[int(x[0])]    #We extract the points of the triangulation containing the i-th remaining point
        A[i,y] = x[1:dim+2]
    targets = np.matmul(A,np.array(trilabels))
    targets = np.array([min(1,math.floor(target/threshold)) for target in targets])

    if real.any()!=None:
        errors = np.abs(targets-real)
        correct = [i for i in range(len(targets)) if errors[i]<=0]
        incorrect = [i for i in range(len(targets)) if errors[i]>0]
        return targets, errors, correct, incorrect
    else:
        return targets
    
def plot_3Ddelaunay(data,labels,sample,rem,tri,ax):
    """
    Plots the data and the triangulation in 3D.
    
    Args:
        - data: data to which add the new points.
        - labels: labels of the data.
        - rem: indices of data not belonging to the triangulation.
        - sample: indices of data belonging to the triangulation.
        - tri: Delaunay triangulation.
        - ax: axes where to plot the data.

    Returns:
        - None
    """
    tri_colors = ['b','r','g']
    for triangle in range(len(tri.simplices)):
        tr = tri.simplices[triangle]
        ll = [labels[rem[i]] for i in range(len(rem)) if tri.find_simplex(data[rem[i]])==triangle]
        if len(ll)!=0:
            color = tri_colors[math.floor(sum(ll)/len(ll))]
            lw = '1'
        else:
            color = 'black'
            lw = '0.5'
        pts = data[sample[tr], :]
        ax.plot3D(pts[[0,1],0], pts[[0,1],1], pts[[0,1],2], color=color, lw=lw, alpha = 0.1)
        ax.plot3D(pts[[0,2],0], pts[[0,2],1], pts[[0,2],2], color=color, lw=lw, alpha = 0.1)
        ax.plot3D(pts[[0,3],0], pts[[0,3],1], pts[[0,3],2], color=color, lw=lw, alpha = 0.1)
        ax.plot3D(pts[[1,2],0], pts[[1,2],1], pts[[1,2],2], color=color, lw=lw, alpha = 0.1)
        ax.plot3D(pts[[1,3],0], pts[[1,3],1], pts[[1,3],2], color=color, lw=lw, alpha = 0.1)
        ax.plot3D(pts[[2,3],0], pts[[2,3],1], pts[[2,3],2], color=color, lw=lw, alpha = 0.1)
    
    return None