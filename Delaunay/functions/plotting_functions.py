import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from classes.Classifier import Classifier
from classes.Measurer import Measurer

def generate_colors2D(labels,points):
    """
    Generates the colors of the points to be plotted in 2D according to their labels.

    Args:
        - labels : labels of the points to be plotted.
        - points : points to be plotted.

    Returns:
        colors : colors of the points to be plotted.
    """
    colors = [[min(1,labels[points][i,0]),0,1-min(labels[points][i,0],1)] for i in range(len(points))]
    colors = [[max(c[0],0),0,max(c[2],0)] for c in colors]
    return colors

def generate_colors3D(labels,points):
    """
    Generates the colors of the points to be plotted in 3D according to their labels.

    Args:
        - labels : labels of the points to be plotted.
        - points : points to be plotted.

    Returns:
        colors : colors of the points to be plotted.
    """
    colors = [[min(1,labels[points][i,0]),min(labels[points][i,1],1),1-min(labels[points][i,0],1)-min(1,labels[points][i,1])] for i in range(len(points))]
    colors = [[max(c[0],0),max(c[1],0),max(c[2],0)] for c in colors]
    return colors

def generate_test_colors(test_labels,dim_labels):
    """
    Generates the colors of the test points according to their labels.

    Args:
        - test_labels :  labels of the test points to be plotted.
        - dim_labels : nº of diferent possible labels - 1.

    Returns:
        test_colors : colors of the test points to be plotted.
    """
    test_colors = []
    if dim_labels == 1:
        for i in range(len(test_labels)):
            if test_labels[i]==0:
                test_colors.append('r')
            else:
                test_colors.append('b')
    else:
        for i in range(len(test_labels)):
            if test_labels[i]==0:
                test_colors.append('r')
            elif test_labels[i] == 1:
                test_colors.append('g')
            else:
                test_colors.append('b')
    return test_colors

def generate_colors(labels,points,dim_labels):
    """
    Generates the colors of the points to be plotted in either 2D or 3D according to their labels.

    Args:
        - labels : labels of the points to be plotted.
        - points : points to be plotted.

    Returns:
        colors : colors of the points to be plotted.
    """
    if dim_labels==1:
        colors = generate_colors2D(labels,points)
    else:
        colors = generate_colors3D(labels,points)
    return colors

def plot_data2D(data,labels,dim_labels,sample,rem,tri,added=[],test_data=[],test_labels=[],correct=[],incorrect=[],write_labels=False):
    """
    Plots the 2D:
        - Classification data (corresponding to the Delaunay triangulation) colored by their estimated labels.
        - Edges of the Delaunay triangulation.
        - Training data colored by their real labels.
        - Test data colored by their real labels and marker corresponding to wheter it has been correctly classified ('o') or not ('x').

    Args:
        - data : data containing classification and training points.
        - labels : labels of data.
        - dim_labels : nº of diferent possible labels - 1.
        - sample : indices of the classification points.
        - rem : indices of training data points.
        - tri : Delaunay triangulation of points from data indexed by sample.
        - added : if not empty, plot the added barycenters bigger.
        - test_data : data used to compute real error.
        - test_labels : labels of test_data.
        - correct : if not empty, indices of points whose estimated label is the same as the real one.
        - incorrect : if not empty, indices of points whose estimated label is not the same as the real one.
        - write_labels : if True, annotate the estimated labels.
    
    Returns:
        fig : resulting 2D plot as pyplot figure.
    """
    fig =  plt.figure(figsize = (10,7))
    ax = fig.add_subplot()

    colors_rem = generate_colors(labels,rem,dim_labels)
    colors_sample = generate_colors(labels,sample,dim_labels)
    ax.scatter(data[rem][:,0],data[rem][:,1],s=2,color=colors_rem,alpha=0.2)
    ax.scatter(data[sample][:,0],data[sample][:,1],s=10,color=colors_sample,alpha=1)
    ax.triplot(data[sample][:,0],data[sample][:,1],tri.simplices,color="black",alpha=0.2,linewidth=0.5)
    if write_labels:
        for i in range(len(sample)):
            ax.annotate(str(round(labels[sample[i],0],2)),[data[sample[i]][0],data[sample[i]][1]])
    ax.scatter(data[added][:,0],data[added][:,1],s=50,color='black')
    if len(test_data)*len(test_labels)!=0:
        test_colors = generate_test_colors(test_labels,dim_labels)
        ax.scatter(test_data[correct][:,0],test_data[correct][:,1],s=100,color = [test_colors[i] for i in correct],marker='o')
        ax.scatter(test_data[incorrect][:,0],test_data[incorrect][:,1],s=100,color = [test_colors[i] for i in incorrect],marker='x')
    return fig

def plot_data3D(data,labels,dim_labels,sample,rem,added=[],test_data=[],test_labels=[],correct=[],incorrect=[],write_labels=False):
    """
    Plots the 3D:
        - Classification data (corresponding to the Delaunay triangulation) colored by their estimated labels.
        - Edges of the Delaunay triangulation.
        - Training data colored by their real labels.
        - Test data colored by their real labels and marker corresponding to wheter it has been correctly classified ('o') or not ('x').

    Args:
        - data : data containing classification and training points.
        - labels : labels of data.
        - dim_labels : nº of diferent possible labels - 1.
        - sample : indices of the classification points.
        - rem : indices of training data points.
        - tri : Delaunay triangulation of points from data indexed by sample.
        - added : if not empty, plot the added barycenters bigger.
        - test_data : data used to compute real error.
        - test_labels : labels of test_data.
        - correct : if not empty, indices of points whose estimated label is the same as the real one.
        - incorrect : if not empty, indices of points whose estimated label is not the same as the real one.
        - write_labels : if True, annotate the estimated labels.
    
    Returns:
        fig : resulting 3D plot as pyplot figure.
    """
    fig =  plt.figure(figsize = (10,7))
    ax = fig.add_subplot(projection='3d')

    colors_rem = generate_colors(labels,rem,dim_labels)
    colors_sample = generate_colors(labels,sample,dim_labels)
    ax.scatter3D(data[rem][:,0],data[rem][:,1],data[rem][:,2],s=2,color=colors_rem)
    ax.scatter3D(data[sample][:,0],data[sample][:,1],data[sample][:,2],s=100,color = colors_sample)
    if write_labels:
        for i in range(len(sample)):
            ax.annotate(str(round(labels[sample[i],0],2)),[data[sample[i]][0],data[sample[i]][1]])
    ax.scatter3D(data[added][:,0],data[added][:,1],data[added][:,2],s=50,color='black')
    if len(test_data)*len(test_labels)!=0:
        test_colors = generate_test_colors(test_labels,dim_labels)
        ax.scatter3D(test_data[correct][:,0],test_data[correct][:,1],test_data[correct][:,2],s=100,color = [test_colors[i] for i in correct],marker='o')
        ax.scatter3D(test_data[incorrect][:,0],test_data[incorrect][:,1],test_data[incorrect][:,2],s=100,color = [test_colors[i] for i in incorrect],marker='x')
    return fig

def plot_data(data,labels,dim,dim_labels,sample,rem,tri,added=[],test_data=[],test_labels=[],correct=[],incorrect=[],write_labels=False):
    """
    Plots the either 2D or 3D:
        - Classification data (corresponding to the Delaunay triangulation) colored by their estimated labels.
        - Edges of the Delaunay triangulation.
        - Training data colored by their real labels.
        - Test data colored by their real labels and marker corresponding to wheter it has been correctly classified ('o') or not ('x').

    Args:
        - data : data containing classification and training points.
        - labels : labels of data.
        - dim_labels : nº of diferent possible labels - 1.
        - sample : indices of the classification points.
        - rem : indices of training data points.
        - tri : Delaunay triangulation of points from data indexed by sample.
        - added : if not empty, plot the added barycenters bigger.
        - test_data : data used to compute real error.
        - test_labels : labels of test_data.
        - correct : if not empty, indices of points whose estimated label is the same as the real one.
        - incorrect : if not empty, indices of points whose estimated label is not the same as the real one.
        - write_labels : if True, annotate the estimated labels.
    
    Returns:
        fig : resulting 2D or 3D plot as pyplot figure.
    """
    if dim==2:
        fig = plot_data2D(data,labels,dim_labels,sample,rem,tri,added,test_data,test_labels,correct,incorrect,write_labels)
    else:
        fig = plot_data3D(data,labels,dim_labels,sample,rem,added,test_data,test_labels,correct,incorrect,write_labels)
    return fig

def plot_scrollable_data(data,labels,sample,rem,dim,dim_labels,it,long_data,long_labels,long_tris,err_dict,run_params):
    """
    Plots scrollable data and metrics. 
    It allows to visualize at each step of the training process:
        - The classification and training points and the Delaunay triangulation.
        - The evolution of the metrics.

    Args:
        - data : data containing classification and training points.
        - labels : labels of data.
        - sample : indices of the classification points.
        - rem : indices of training data points.
        - dim : dimension of data.
        - dim_labels : nº of diferent possible labels - 1.
        - it : number of times to move points, estimate labels and compute metrics.
        - long_data : trajectories of the classification points
        - long_labels : estimated labels of the classification points at each time.
        - long_tris : Delaunay triangulations at each time
        - err_dict : dictionary containing the metrics of the training.
        - run_params : parameters that characterize how the data has been initialized and trained. 

    Returns:
        - None
    """
    errw = run_params['errw']
    avw = run_params['avw']
    colors_rem = generate_colors(labels,rem,dim_labels)
    colors_sample = generate_colors(long_labels[0],range(len(long_labels[0])),dim_labels)
    fig =  plt.figure(figsize = (10,7))
    fig.suptitle('Error weight = '+str(errw)+'; average weight = '+str(avw))
    if dim==2:
        ax0 = fig.add_subplot(121)
        ax0.scatter(long_data[0][:,0],long_data[0][:,1],color=colors_sample)
        ax0.scatter(data[rem][:,0],data[rem][:,1],s=2,color=colors_rem, alpha=0.2)
    else:
        ax0 = fig.add_subplot(121,projection='3d')
        ax0.scatter3D(long_data[0][:,0],long_data[0][:,1],long_data[0][:,2],color=colors_sample)
        ax0.scatter3D(data[rem][:,0],data[rem][:,1],data[rem][:,2],s=2,color=colors_rem, alpha=0.2)
    ax1 = fig.add_subplot(122)
    axcolor = "White"
    time_axis = plt.axes([0.20, 0.05, 0.65, 0.03], 
                            facecolor = axcolor)
    time_slider2 = Slider(time_axis,'Time',0,it-1,valinit=0)

    def update2(val):
        val = int(val)
        data_val, labels_val, tri_val = long_data[val], long_labels[val], long_tris[val]
        ax0.clear()
        ax1.clear()
        ax1.set_title('Training error')
        if dim==2:
            ax0.set_title('xy')
            colors_sample = generate_colors(labels_val,range(len(labels_val)),dim_labels)
            ax0.scatter(data_val[:,0],data_val[:,1],color=colors_sample,s=10,alpha=1)
            ax0.scatter(data[rem][:,0],data[rem][:,1],s=2,color=colors_rem, alpha=0.2)
            """ for i in range(len(long_data[int(val)])):
                if round(long_labels[int(val)][i,0],3)!=0.0 and round(long_labels[int(val)][i,0],3)!=1.0:
                    text = str(round(long_labels[int(val)][i,0],3))
                    ax20.text(long_data[int(val)][i,0],long_data[int(val)][i,1],text) """
            ax0.triplot(data_val[:,0],data_val[:,1],tri_val.simplices,color = "black",alpha=0.2,linewidth=0.5)
            ax1.plot(range(val+1),err_dict['training_error'][:(val+1)],'--+')
            ax1.plot(range(val+1),err_dict['real_error'][:(val+1)],'--+')
        else:
            ax0.set_title('xyz')
            colors_sample = generate_colors(labels_val,range(len(labels_val)),dim_labels)
            ax0.scatter3D(data_val[:,0],data_val[:,1],data_val[:,2],color=colors_sample,s=10,alpha=1)
            ax0.scatter3D(data[rem][:,0],data[rem][:,1],data[rem][:,2],s=2,color=colors_rem, alpha=0.2)
            """ for i in range(len(long_data[int(val)])):
                if round(long_labels[int(val)][i,0],3)!=0.0 and round(long_labels[int(val)][i,0],3)!=1.0:
                    text = str(round(long_labels[int(val)][i,0],3))
                    ax20.text(long_data[int(val)][i,0],long_data[int(val)][i,1],text) """
            ax1.plot(range(val+1),err_dict['training_error'][:(val+1)],'--+')
            ax1.plot(range(val+1),err_dict['real_error'][:(val+1)],'--+')
    time_slider2.on_changed(update2)
    plt.show()

def plot_classifier(Classifier: Classifier):
    """
    Plots the data of a classifier.

    Args:
        - Classifier : Classifier the points of which are to be plotted.

    Returns:
        - fig : plot of the classifier as a pyplot figure.
    """
    fig = plot_data(Classifier.data,Classifier.labels,Classifier.dim,Classifier.dim_labels,Classifier.sample,Classifier.rem,Classifier.tri)
    return fig
    
def plot_classifier_scrollable(Classifier: Classifier,it,long_data,long_labels,long_tris,err_dict,run_params):
    """
    Plots a scrollable plot of the data of a classifier during its training.

    Args:
        - Classifier : Classifier the points of which are to be plotted.
        - it : number of times to move points, estimate labels and compute metrics.
        - long_data : trajectories of the classification points
        - long_labels : estimated labels of the classification points at each time.
        - long_tris : Delaunay triangulations at each time
        - err_dict : dictionary containing the metrics of the training.
        - run_params : parameters that characterize how the data has been initialized and trained. 

    Returns:
        - None
    """
    plot_scrollable_data(Classifier.data,Classifier.labels,Classifier.sample,Classifier.rem,Classifier.dim,Classifier.dim_labels,it,long_data,long_labels,long_tris,err_dict,run_params)

def plot_metrics(Measurer: Measurer):
    """
    Plots metrics of the training of a classifier.

    Args:
        - Measurer : Measurer containing the metrics of the training of a classifier.

    Returns:
        - fig : plot of the metrics as a pyplot figure.
    """
    fig, ax = plt.subplots(2,2)
    metrics = ['training_error','error_variance','edge_variance','real_error']
    for i in range(len(metrics)):
        metric = metrics[i]
        ax[i//2,i%2].plot(range(len(Measurer.metrics[metric])),Measurer.metrics[metric],'--+')
        ax[i//2,i%2].set_title(metric)
    
    return fig