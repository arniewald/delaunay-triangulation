import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from time import time
from qpsolvers import solve_qp

from functions.initialization_functions import subtesselate
from functions.measuring_functions import adjacency, mean_training_error

from classes.Classifier import Classifier
from classes.Measurer import Measurer

def compute_eq_parameters(Classifier: Classifier, avw):
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

    W_shape = ((len(Classifier.rem)+len(Classifier.sample))*Classifier.dim_labels,(len(Classifier.rem)+len(Classifier.sample))*Classifier.dim_labels)
    #dim_labels = nº of possible labels - 1
    #B = labels[:,0]
    Classifier.rem = [i for i in range(len(Classifier.data)) if i not in Classifier.sample]
    B = Classifier.labels[Classifier.rem,0]
    for i in range(1,Classifier.dim_labels):
        #B = np.concatenate([B,labels[:,i]])
        B = np.concatenate([B,Classifier.labels[Classifier.rem,i]])
    
    
    print('\tBuilding barycentric coordinates equations...')
    M = np.zeros((len(Classifier.rem),len(Classifier.sample)), dtype=np.float32)
    for i in range(len(Classifier.rem)):
        x = Classifier.bc[Classifier.rem[i]][1:(Classifier.dim+3)]            #We extract index and barycentric coordinates of the i-th remaining point
        y = Classifier.tri.simplices[int(x[0])]    #We extract the points of the triangulation containing the i-th remaining point
        M[i,y] = x[1:Classifier.dim+2]
    A = block_diag(*([M]*Classifier.dim_labels))
    print('\tBarycentric coordinates equations built')

    print('\tBuilding averages equations...')
    adj = adjacency(Classifier.tri,range(len(Classifier.sample)))
    print('\t \t Adjacency built')
    S = np.zeros((len(Classifier.sample),len(Classifier.sample)), dtype=np.float32)
    for i in range(len(Classifier.sample)):
        nodes = [node for node in adj[i] if node != i]
        len_nodes = len(nodes)
        S[i,i] = -len_nodes
        for node in nodes:
            S[i,node] = 1
    A = np.concatenate([A,block_diag(*([S]*Classifier.dim_labels))])
    B = np.concatenate([B,np.zeros(len(Classifier.sample)*Classifier.dim_labels)])
    print('\tAverages equations built')

    print('\tBuilding weighting matrix...')
    W = np.zeros(W_shape)
    for i in range(len(Classifier.rem)*Classifier.dim_labels):
        W[i,i] = 1
    for i in range(len(Classifier.rem)*Classifier.dim_labels, (len(Classifier.rem)+len(Classifier.sample))*Classifier.dim_labels):
        W[i,i] = avw
    print('\tWeighting matrix built')

    return A, B, M, S, W

def add_barycenters(Classifier: Classifier, e):
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
    tri_errors = mean_training_error(Classifier, e)
    d = dict()
    for i in range(len(tri_errors)-20,len(tri_errors)):
        key = list(tri_errors.keys())[i]
        d[key] = tri_errors[key]
    tri_errors = d
    added = []
    for triangle in tri_errors.keys():
        #if tri_errors[triangle][1] > mte_threshold:
            Classifier.data = np.concatenate([Classifier.data,np.array([tri_errors[triangle][0]])])
            new_label = np.sum([Classifier.labels[Classifier.sample[i],:] for i in Classifier.tri.simplices[triangle]], axis = 0)/(Classifier.dim+1)
            Classifier.labels = np.concatenate([Classifier.labels,np.array([new_label])])
            Classifier.sample = np.concatenate([Classifier.sample,np.array([int(len(Classifier.data)-1)])])
            Classifier.out_hull = np.concatenate([Classifier.out_hull,np.array([int(len(Classifier.sample)-1)])])
            added.append(Classifier.sample[-1])
    
    return added

def delaunayization(Classifier: Classifier,avw):
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
    Classifier.tri,  Classifier.bc = subtesselate(Classifier.data, Classifier.sample, Classifier.dim)
    start = time()
    A, B, M, S, W = compute_eq_parameters(Classifier, avw)
    
    #To minimize: ||Ax-B||^2
    #Minimize 0.5*xT*P*x+qT*x s. t. Gx<=h, lb<=x<=ub
    #P = 2*AT*A, q=-2*AT*B
    At = np.transpose(A)
    P = csc_matrix(2*np.matmul(np.matmul(At,W),A))
    q = -2*np.matmul(np.matmul(At,W),B)
    
    #We define the constraint matrix. We want: C*y = y1 + y2 <= np.ones(len(sample))
    G = np.zeros((len( Classifier.sample),len(Classifier.sample)*Classifier.dim_labels), dtype = np.float32)
    for i in range(len(Classifier.sample)):
        for j in range(Classifier.dim_labels):
            G[i,i+j*len(Classifier.sample)] = 1
    h = np.ones(len(Classifier.sample))
    G = csc_matrix(G)
    #lb = 0, ub = 1
    lb = np.zeros(len(Classifier.sample)*Classifier.dim_labels)
    ub = np.ones(len(Classifier.sample)*Classifier.dim_labels)
    end = time()
    print('Time to build parameters: '+str(end-start))

    start = time()
    print('Starting lsq fit...')
    y_aux = solve_qp(P,q,G=G,h=h,A=None,b=None,lb=lb,ub=ub,solver='piqp')
    end = time()
    print('Time to solve lsqfit: '+str(end-start))

    #This is extremly unoptimal but I don't know how to do it better right now: from matrix to column, from column to matrix.
    len_rem = len(Classifier.data)-len(Classifier.sample)
    y = np.zeros((len(Classifier.sample),Classifier.dim_labels))
    for i in range(Classifier.dim_labels):
        y[:,i] = y_aux[i*len(Classifier.sample):(i+1)*len(Classifier.sample)]
    e = abs(np.matmul(A[:len_rem*Classifier.dim_labels,:],y_aux)-B[:len_rem*Classifier.dim_labels])
    
    
    e_aux = np.zeros((len_rem,Classifier.dim_labels))
    for i in range(Classifier.dim_labels):
        e_aux[:,i] = e[i*len_rem:(i+1)*len_rem]
    e = np.array([np.sqrt(sum(e_aux[i]**2)) for i in range(len(e_aux))])
    
    start = time()
    Mt = np.transpose(M)
    P2 = csc_matrix(2*np.matmul(Mt,M))
    q2 = -2*np.matmul(e,M)
    lb = np.zeros(len(Classifier.sample))
    ub = np.ones(len(Classifier.sample))
    err = solve_qp(P2,q2,G=None,h=None,A=None,b=None,lb=lb,ub=ub,solver='piqp')
    print('Time to solve quadratic: '+str(end-start))

    Classifier.labels[Classifier.sample,:] = y
    return e, err

def train_step(Classifier: Classifier,al,errw,err):
    start = time()
    L = len(Classifier.out_hull)
    adj = adjacency(Classifier.tri,Classifier.out_hull)
    disin = np.zeros(L)
    errin = np.zeros(L)
    for i in range(L):
        try:
            errin[i] = Classifier.sample[adj[Classifier.out_hull[i]][np.argmax([(err[j]-err[Classifier.out_hull[i]]) for j in adj[Classifier.out_hull[i]]])]]
            disin[i] = Classifier.sample[adj[Classifier.out_hull[i]][np.argmax([sum((Classifier.data[Classifier.sample[j]]-Classifier.data[Classifier.sample[Classifier.out_hull[i]]])**2) for j in adj[Classifier.out_hull[i]]])]]
        except Exception as e:
            print("Exception at node ",Classifier.sample[i],": ",adj[i])
            print("Exception: ",e)
    errin = [int(i) for i in errin]
    disin = [int(i) for i in disin]
    Classifier.data[Classifier.sample[Classifier.out_hull]] += al*(errw*(Classifier.data[errin]-Classifier.data[Classifier.sample[Classifier.out_hull]])+(1-errw)*(Classifier.data[disin]-Classifier.data[Classifier.sample[Classifier.out_hull]]))
    end = time()
    print('Time to move points: '+str(end-start))
    return None


