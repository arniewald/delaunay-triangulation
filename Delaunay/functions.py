import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn
from scipy.spatial import Delaunay, ConvexHull
from scipy.optimize import lsq_linear, nnls
import random
from sklearn.datasets import make_circles, make_classification, make_moons
from time import time

def f(x,y):
    """
    Example function to give values to the points.
    """
    return (1+np.sign(np.sin(np.sqrt(x*x+y*y))))/2

def initialize_sample_old(data,size,dim):
    length = len(data)
    hull = ConvexHull(data)
    hull = list(hull.vertices)
    sample = random.sample([i for i in range(length) if i not in hull],size)
    sample = np.concatenate([sample,hull])
    sample.sort()
    rem = [i for i in range(length) if i not in sample]
    return np.array(sample), np.array(rem), hull

def refine(data,dim,rep=1):
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
    old_data = data
    return old_data, new_data, in_hull, L

def initialize_sample(data,size,dim,rep=1):
    old_data, new_data, in_hull, L = refine(data,dim,rep)
    length = len(new_data)
    sample = random.sample([int(i) for i in range(L,length)],size)
    sample = np.concatenate([in_hull,sample])
    sample.sort()
    sample = [int(i) for i in sample]
    rem = [int(i) for i in range(length) if int(i) not in sample]
    return old_data, new_data, np.array(sample), np.array(rem), in_hull

def adj_matrix(tri, sample, in_hull): #NEEDS OPTIMIZATION
    L = len(sample)
    adj = np.zeros((L,L),dtype=int)
    for triangle in tri.simplices:
        i, j, k = triangle[0], triangle[1], triangle[2]
        if sample[i] in in_hull:
            adj[i,i] = 2
        if sample[j] in in_hull:
            adj[j,j] = 2
        if sample[k] in in_hull:
            adj[k,k] = 2
        adj[i,j] = 1
        adj[j,i] = 1
        adj[i,k] = 1
        adj[k,i] = 1
        adj[j,k] = 1
        adj[k,j] = 1
    return adj

def subtesselate(data,sample,rem,dim):
    
    """ 
    Builds the Delaunay triangulation from a subsample of points containing the convex hull and computes the barycentric coordinates from the other points.
    
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
        to_append = np.concatenate([[int(i),int(triangle)],c])
        bc.append(np.concatenate([[int(i),int(triangle)],c]))

    return tri, bc

def calc_bc(tri, triangle, point):
    
    """
    Calculates the barycentric coordinates of a point with respect to a triangle from a triangulation.

    Args:
        - tri: triangulation from which the triangle comes.
        - triangle: vertices from which the barycentric coordinates are calculated.
        - point: point of which to calculate the barycentric coordinates.

    Returns:
        - c: barycentric coordinates of point with respect to triangle.
    """

    b = tri.transform[triangle,:2].dot(np.transpose(point - tri.transform[triangle,2]))
    c = np.concatenate([b,[1-sum(b)]])
    return c

def estimate_height(tri, point, labels):

    """
    Calculates the height/value/label of a point given the Delaunay triangulation and the labels of their points.

    Args:
        - tri: Delaunay triangulation.
        - point: point from which to estimate the value.
        - labels: values of the points of the triangulation.

    Returns:
        - label: estimated value of point.
    """

    triangle = tri.find_simplex(point)
    verts = tri.simplices[triangle]
    bc = calc_bc(tri, triangle, point)
    label = labels[verts[0]]*bc[0]+labels[verts[1]]*bc[1]+labels[verts[2]]*bc[2]
    return label

def compute_eq_parameters(data, tri, rem, sample, bc, dim):
    B = data[rem][:,dim]
    A = np.zeros((len(rem),len(sample)))
    for i in range(len(rem)):
        s, x, y, z = bc[i][1:5]
        a, b, c = tri.simplices[int(s)]
        A[i][a], A[i][b], A[i][c] = x, y, z
    return A, B
                
def movepoints_step(data, sample, err, adj, al, errw=0.5): 
    start = time()
    L = len(sample)
    incrs = np.zeros((L,2))
    for i in range(L):
        if adj[i,i] == 0: #Check that the point is not in the convex hull
            k1, k2 = i, i
            diferr = 0
            difdis = 0
            for j in range(L):
                if adj[i,j] == 1:
                    difauxerr = err[j]-err[i]
                    difauxdis = (data[sample[i]][0]-data[sample[j]][0])**2+(data[sample[i]][1]-data[sample[j]][1])**2
                    if difauxerr>diferr:
                        k1 = j
                        diferr = difauxerr
                    if difauxdis>difdis:
                        k2 = j
                        difdis = difauxdis
            indexi = sample[i]
            if k1 != i:
                indexk1 = sample[k1]
                incrs[i,0] += al*errw*(data[indexk1][0]-data[indexi][0])
                incrs[i,1] += al*errw*(data[indexk1][1]-data[indexi][1])
            indexk2 = sample[k2]
            incrs[i,0] += al*(1-errw)*(data[indexk2][0]-data[indexi][0])
            incrs[i,1] += al*(1-errw)*(data[indexk2][1]-data[indexi][1])

    for i in range(L):
        data[sample[i]][0] += incrs[i][0]
        data[sample[i]][1] += incrs[i][1]
    end = time()
    print('Time to move points: '+str(end-start))
    return None

def delaunayization(data,sample,rem,in_hull,dim,lb=-np.inf,ub=np.inf,binary=False,threshold=0.5):
    tri, bc = subtesselate(data[:,0:dim],sample,rem,dim)
    adj = adj_matrix(tri,sample,in_hull)
    A, B = compute_eq_parameters(data, tri, rem, sample, bc, dim)
    P = 2*np.matmul(A.transpose(),A)

    #Linear problem
    #sol = np.linalg.lstsq(A,B,rcond=None)
    start = time()
    sol = lsq_linear(A,B,bounds=(0,1))
    end = time()
    print('Time to solve lsqfit: '+str(end-start))
    y = sol['x']
    if binary == True:
        y = (1+np.sign(y-threshold))/2
    print((y.all() >= lb and y.all() <= ub))
    e = abs(np.matmul(A,y)-B)

    #Quadratic problem (||Ax-b||^2=0,x>=0)
    start = time()
    err = nnls(A,e)[0]
    end = time()
    print('Time to solve quadratic: '+str(end-start))

    return tri, sample, rem, adj, bc, y, e, err

def movepoints(data,sample,rem,in_hull,dim,time,al,errw=0.5,lb=-np.inf,ub=np.inf,binary=False,threshold=0.5):
    esterrs, lsqferrs, sigmas, rerrs = [], [], [], []
    for i in range(time):
        try:
            tri, sample, rem, adj, bc, y, e, err = delaunayization(data,sample,rem,in_hull,dim,lb,ub,binary,threshold)
            print(max(y))
            print(min(y))
            errav = sum(err)/len(sample)
            sigma = 0
            for j in range(len(err)):
                sigma += err[j]*err[j]
            sigma = np.sqrt(sigma/len(err) - errav*errav)
            esterr = sum(err)
            lsqferr = sum(e)
            rerr = 0
            for j in range(len(y)):
                rerr += abs(y[j]-f(data[sample[j]][0],data[sample[j]][1]))
            rerrs.append(rerr)
            esterrs.append(esterr)
            lsqferrs.append(lsqferr)
            sigmas.append(sigma)

            print(i,esterr,lsqferr,sigma,rerr)
            

            movepoints_step(data, sample, err, adj, al,errw)

        except Exception as e:
            print(e)

    tri, sample, rem, adj, bc, y, e, err = delaunayization(data,sample,rem,in_hull,dim,lb,ub,binary,threshold)
    fsigmas = open("sigmas.txt","w")
    festerrs = open("esterrs.txt","w")
    flsqferrs = open("lsqferrs.txt","w")
    frerrs = open("rerrs.txt","w")
    for i in range(time-1):
        fsigmas.write(str(sigmas[i])+"\t")
        festerrs.write(str(esterrs[i])+"\t")
        flsqferrs.write(str(lsqferrs[i])+"\t")
        frerrs.write(str(rerrs[i])+"\t")
    fsigmas.close()
    festerrs.close()
    flsqferrs.close()
    frerrs.close()

    return tri, sample, rem, adj, bc, y, e, err