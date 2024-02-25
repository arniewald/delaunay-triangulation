import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.optimize import lsq_linear, nnls
import random
from time import time

def f(x,y):
    """
    Example function to give values to the points.
    """
    return np.exp(-x*x)*np.sin(y)

def refine(data,dim,rep=1):
    """
    Adds points to the convex hull.

    Args:
        - data: data to which add the new points.
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
    hull = ConvexHull(data[:,0:dim])
    hull = list(hull.vertices)
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
    return new_data, in_hull, L

def initialize_sample(data,size,dim,rep=1):
    """
    Refines the convex hull of data and selects sample to perform Delaunay triangulation.

    Args:
        - data: array with initial data.
        - size: size of the desired sample. If convex hull is bigger, the sample will just contain the convex hull.
        - dim: dimension of data: each element of data has dim features and one label.
        - rep: number of times to refine the convex hull.
    
    Returns:
        - new data: data reordered (first elements belong to sample) and with refinement of the complex hull.
        - sample: labels of the data selected for the sample.
        - rem: labels of the data not selected for the sample.
        - out_hull: labels of the elements of sample not belonging to the convex hull.
    """
    new_data, in_hull, L = refine(data,dim,rep)
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
    return new_data, sample, rem, out_hull

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

def subtesselate(data,sample,rem,dim):
    
    """ 
    Builds the Delaunay triangulation from a subsample of points containing the convex hull 
    and computes the barycentric coordinates from the other points.
    
    Args: 
        - data: original set of points.
        - sample: points from which to compute the Delaunay triangulation.
        - rem: points from which to compute the barycentric coordinates.
        - dim: dimension of the points of data.
    
    Returns:
        - tri: Delaunay triangulation. The indexation of the vertices comes from sample, not from data.
        - bc: barycentric coordinates of rem with respect to tri.
    """
    tri = Delaunay(data[sample])
    bc = []
    
    for i in rem:
        point = data[i]
        triangle = tri.find_simplex(point)
        b = tri.transform[triangle,:dim].dot(np.transpose(point - tri.transform[triangle,dim]))
        c = np.concatenate([b,[1-sum(b)]])
        bc.append(np.concatenate([[int(i),int(triangle)],c])) #Resulting array: index of point, index of triangle and barycentric coordinates
    return tri, bc


def compute_eq_parameters(data, tri, rem, sample, bc, dim):
    """
    Computes the matrix and the column from which to estimate the labels of the points of the triangulation.

    Args:
        - data: array with data.
        - tri: Delaunay triangulation.
        - rem: labels of data not belonging to the triangulation.
        - sample: labels of data belonging to the triangulation.
        - bc: barycentric coordinates of the points indexated by rem.
        - dim: dimension of data.
    
    Returns:
        - A, B: the matrices that describe the least squares problem: ||Ax-B||**2=0.
    """
    B = data[rem][:,dim]
    A = np.zeros((len(rem),len(sample)))
    for i in range(len(rem)):
        x = bc[i][1:(dim+3)]            #We extract index and barycentric coordinates of the i-th remaining point
        y = tri.simplices[int(x[0])]    #We extract the points of the triangulation containing the i-th remaining point
        for j in range(dim+1):
            A[i][y[j]] = x[j+1]
    return A, B
                
def movepoints_step(data, sample, out_hull, tri, err, dim, al, errw=0.5):
    """
    Moves one time the points of sample not in the convex hull according to the error and the distance gradient.

    Args:
        - data: array with data.
        - sample: labels of data belonging to the triangulation.
        - out_hull: labels of the elements of sample not belonging to the convex hull.
        - tri: Delaunay triangulation.
        - err: estimated errors of the points of the triangulation.
        - dim: dimension of data.
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
        errin[i] = sample[adj[out_hull[i]][np.argmax([(err[j]-err[out_hull[i]]) for j in adj[out_hull[i]]])]]
        disin[i] = sample[adj[out_hull[i]][np.argmax([sum((data[sample[j]][0:dim]-data[sample[out_hull[i]]][0:dim])**2) for j in adj[out_hull[i]]])]]
    errin = [int(i) for i in errin]
    disin = [int(i) for i in disin]
    data[sample[out_hull]] += al*(errw*(data[errin]-data[sample[out_hull]])+(1-errw)*(data[disin]-data[sample[out_hull]]))
    end = time()
    print('Time to move points: '+str(end-start))
    return None

    

def delaunayization(data,sample,rem,dim,lb=-np.inf,ub=np.inf,binary=False,threshold=0.5):
    """
    Performs Delaunay triangulation, computes barycentric coordinates, estimates labels and estimates error.

    Args:
        - data: array with data.
        - sample: labels of data belonging to the triangulation.
        - rem: labels of data not belonging to the triangulation.
        - dim: dimension of data.
        - lb: lower boundary of estimated labels.
        - ub: upper boundary of estimated labels.
        - binary: if True, the data is mapped to 0 if lower than threshold and to 1 if greater.
        - threshold: value from which map binary labels.
    
    Returns:
        - tri: Delaunay triangulation.
        - e: array with least square residuals.
        - err: array with estimated errors.
    """
    tri, bc = subtesselate(data[:,0:dim],sample,rem,dim)
    A, B = compute_eq_parameters(data, tri, rem, sample, bc, dim)

    #Linear problem
    start = time()
    #When there are no bounds, there are values that explode; in this case, since A is sparse (dim+1<<len(sample)), we set lsq_solver='lsmr'
    if lb>-np.inf and ub<np.inf:
        lsq_solver = None
    else:
        lsq_solver='lsmr'
    sol = lsq_linear(A,B,bounds=(lb,ub),lsq_solver=lsq_solver)
    end = time()
    print('Time to solve lsqfit: '+str(end-start))
    y = sol['x']
    if binary == True:
        y = (1+np.sign(y-threshold))/2
    e = abs(np.matmul(A,y)-B)
    data[sample][:,dim] = y

    #Quadratic problem (||Ax-b||^2=0,x>=0)
    start = time()
    err = nnls(A,e)[0] #Faster than lsq_linear
    end = time()
    print('Time to solve quadratic: '+str(end-start))
    return tri, e, err

def movepoints(data,sample,rem,out_hull,dim,it,al,errw=0.5,lb=-np.inf,ub=np.inf,binary=False,threshold=0.5):
    """
    Performs the estimation of labels and movement of points as many times as indicated.
    Also writes, for each iteration: the sum of estimated errors, the sum of the squared residuals, the variance of the estimated
    error and the real error (only if function f applied).

    Args:
        - data: array with data.
        - sample: labels of data belonging to the triangulation.
        - rem: labels of data not belonging to the triangulation.
        - out_hull: labels of the elements of sample not belonging to the convex hull.
        - dim: dimension of data.
        - it: number of times to move the points.
        - al: measures the magnitude of the overall displacement.
        - errw: weight given to the error gradient (weight given to the distance gradient is 1-errw).
        - lb: lower boundary of estimated labels.
        - ub: upper boundary of estimated labels.
        - binary: if True, the data is mapped to 0 if lower than threshold and to 1 if greater.
        - threshold: value from which map binary labels.

    Returns:
        - tri: final Delaunay triangulation.
        - e: final residuals of least squares.
        - err: final estimated errors.
    """
    esterrs, lsqferrs, sigmas, rerrs = [], [], [], []
    for i in range(it):
        try:
            tri, e, err = delaunayization(data,sample,rem,dim,lb,ub,binary,threshold)
            errav = sum(err)/len(sample)
            sigma = 0
            for j in range(len(err)):
                sigma += err[j]*err[j]
            sigma = np.sqrt(sigma/len(err) - errav*errav)
            esterr = sum(err)
            lsqferr = sum(e)
            rerr = 0
            for j in range(len(sample)):
                rerr += abs(data[sample[j]][dim]-f(data[sample[j]][0],data[sample[j]][1]))
            rerrs.append(rerr)
            esterrs.append(esterr)
            lsqferrs.append(lsqferr)
            sigmas.append(sigma)

            print(i,esterr,lsqferr,sigma,rerr)
            

            movepoints_step(data, sample, out_hull, tri, err, dim, al, errw)

        except Exception as e:
            print(e)

    tri, e, err = delaunayization(data,sample,rem,dim,lb,ub,binary,threshold)
    fsigmas = open("sigmas.txt","w")
    festerrs = open("esterrs.txt","w")
    flsqferrs = open("lsqferrs.txt","w")
    frerrs = open("rerrs.txt","w")
    for i in range(it-1):
        fsigmas.write(str(sigmas[i])+"\t")
        festerrs.write(str(esterrs[i])+"\t")
        flsqferrs.write(str(lsqferrs[i])+"\t")
        frerrs.write(str(rerrs[i])+"\t")
    fsigmas.close()
    festerrs.close()
    flsqferrs.close()
    frerrs.close()

    return tri, e, err