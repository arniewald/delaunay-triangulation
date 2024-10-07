import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import Delaunay, ConvexHull
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from time import time
from qpsolvers import solve_qp

from functions.reading_functions import extract_run_params, read_circles_data
from functions.initialization_functions import refine
from functions.measuring_functions import adjacency
from functions.class_functions import premeasurement, fully_train
from functions.plotting_functions import plot_classifier, plot_classifier_scrollable, plot_metrics
from classes.Classifier import *
from classes.Trainer import Trainer
from classes.Measurer import Measurer
from classes.Writer import Writer
from classes.Reader import Reader

"""
Distance gradient
Refining
Barycenters
Average weight
Entropic sampling
"""

def mean_training_error(tri, e):
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
    
    out_hull = [int(i) for i in out_hull]
    return added

data_name = 'circles'
size = 100
al = 0.3
errw = 1

data, labels, dim = read_circles_data(n_samples = 10000, noise = 0.1)

hull = ConvexHull(data)
hull = list(hull.vertices)

length = len(hull)
out_hull_size = max(0,size-length)
out_hull = random.sample([i for i in range(len(data)) if i not in hull],out_hull_size)
sample = np.concatenate([hull,out_hull])
sample = np.array([int(i) for i in sample])
sample.sort()
out_hull = [i for i in range(len(sample)) if sample[i] in out_hull]
rem = np.array([int(i) for i in range(len(data)) if int(i) not in sample])
tri, bc = subtesselate(data,sample,dim)

B = labels[rem].copy()
A = np.zeros((len(rem),len(sample)), dtype=np.float32)
for i in range(len(rem)):
    x = bc[rem[i]][1:(dim+3)]            #We extract index and barycentric coordinates of the i-th remaining point
    y = tri.simplices[int(x[0])]    #We extract the points of the triangulation containing the i-th remaining point
    A[i,y] = x[1:(dim+2)]

At = np.transpose(A)
print(B.shape,At.shape)
P = csc_matrix(2*np.matmul(At,A))
q = -2*np.matmul(At,B)
lb = np.zeros(len(sample))
ub = np.ones(len(sample))
print('Starting lsq fit...')
labels[sample] = solve_qp(P,q,lb=lb,ub=ub,solver='piqp')


e = abs(np.matmul(A[:len(rem),:],labels[sample])-B[:len(rem)])
P2 = P
q2 = -2*np.matmul(At,e)
err = solve_qp(P2,q2,lb=lb,solver='piqp')

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
d = dict()
for i in range(len(tri_errors)-20,len(tri_errors)):
    key = list(tri_errors.keys())[i]
    d[key] = tri_errors[key]
tri_errors = d
facecolors=[1 for _ in range(len(tri.simplices))]
for triangle in tri_errors.keys():
    facecolors[triangle] = 0.5

fig2, ax2 = plt.subplots()
colors_sample = [[min(1,labels[i]),0,1-min(labels[i],1)] for i in sample]
colors_sample = [[max(c[0],0),0,max(c[2],0)] for c in colors_sample]
colors_rem = [[min(1,labels[i]),0,1-min(labels[i],1)] for i in rem]
colors_rem = [[max(c[0],0),0,max(c[2],0)] for c in colors_rem]
sizes = np.ones(len(err))+100*err
ax2.scatter(data[rem][:,0],data[rem][:,1],s=2,color=colors_rem,alpha=0.2)
ax2.scatter(data[sample][:,0],data[sample][:,1],s=sizes,color=colors_sample,alpha=1)

ax2.triplot(data[sample][:,0],data[sample][:,1],tri.simplices,color="black",alpha=1,linewidth=0.5)
ax2.tripcolor(data[sample][:,0],data[sample][:,1],tri.simplices,facecolors=facecolors,alpha=0.2)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_title('Before adding barycenters')



added = []
for triangle in tri_errors.keys():
    data = np.concatenate([data,np.array([tri_errors[triangle][0]])])
    new_label = np.sum([labels[sample[i]] for i in tri.simplices[triangle]], axis = 0)/(dim+1)
    labels = np.concatenate([labels,np.array([new_label])])
    sample = np.concatenate([sample,np.array([int(len(data)-1)])])
    out_hull = np.concatenate([out_hull,np.array([int(len(sample)-1)])])
    added.append(sample[-1])

out_hull = [int(i) for i in out_hull]
not_added = [i for i in sample if i not in added]

fig3, ax3 = plt.subplots()
colors_added = [[min(1,labels[i]),0,1-min(labels[i],1)] for i in added]
colors_added = [[max(c[0],0),0,max(c[2],0)] for c in colors_added]
ax3.scatter(data[rem][:,0],data[rem][:,1],s=2,color=colors_rem,alpha=0.1)
ax3.scatter(data[not_added][:,0],data[not_added][:,1],s=sizes,color=colors_sample,alpha=0.3)
ax3.scatter(data[added][:,0],data[added][:,1],color=colors_added,alpha=1, marker='x')

ax3.triplot(data[not_added][:,0],data[not_added][:,1],tri.simplices,color="black",alpha=1,linewidth=0.5)

ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.set_title('After adding barycenters')

plt.show()