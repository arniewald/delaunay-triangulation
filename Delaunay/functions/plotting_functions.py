import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from classes.Classifier import Classifier
from classes.Measurer import Measurer

def generate_colors2D(labels,points):
    colors = [[min(1,labels[points][i,0]),0,1-min(labels[points][i,0],1)] for i in range(len(points))]
    colors = [[max(c[0],0),0,max(c[2],0)] for c in colors]
    return colors

def generate_colors3D(labels,points):
    colors = [[min(1,labels[points][i,0]),min(labels[points][i,1],1),1-min(labels[points][i,0],1)-min(1,labels[points][i,1])] for i in range(len(points))]
    colors = [[max(c[0],0),max(c[1],0),max(c[2],0)] for c in colors]
    return colors

def generate_test_colors(test_labels,dim_labels):
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
    if dim_labels==1:
        colors = generate_colors2D(labels,points)
    else:
        colors = generate_colors3D(labels,points)
    return colors

def plot_data2D(data,labels,dim_labels,sample,rem,tri,added=[],test_data=[],test_labels=[],correct=[],incorrect=[],write_labels=False):
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
    if dim==2:
        fig = plot_data2D(data,labels,dim_labels,sample,rem,tri,added,test_data,test_labels,correct,incorrect,write_labels)
    else:
        fig = plot_data3D(data,labels,dim_labels,sample,rem,added,test_data,test_labels,correct,incorrect,write_labels)
    return fig

def plot_scrollable_data(data,labels,sample,rem,dim,dim_labels,it,long_data,long_labels,long_tris,err_dict,run_params):
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

def plot_errors(data_name, filename):
    path = str(os.getcwd())+'\\results\\' + data_name + '\\errors\\' + filename + '.csv'
    df = pd.read_csv(path)
    err_dict = dict()
    columns = ['avs','sigmas','evars','rerrs']
    for column in columns:
        err_dict[column] = df[column].tolist()

    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(range(len(err_dict['avs'])),err_dict['avs'],'--+')
    ax[0,0].set_title('Average training error')

    ax[0,1].plot(range(len(err_dict['sigmas'])),err_dict['sigmas'],'--+')
    ax[0,1].set_title('Training error standard deviation')

    ax[1,0].plot(range(len(err_dict['evars'])),err_dict['evars'],'--+')
    ax[1,0].set_title('Edge length variance')

    ax[1,1].plot(range(len(err_dict['rerrs'])),err_dict['rerrs'],'--+')
    ax[1,1].set_title('Average real error')
    return fig

def plot_classifier(Classifier: Classifier):
    plot_data(Classifier.data,Classifier.labels,Classifier.dim,Classifier.dim_labels,Classifier.sample,Classifier.rem,Classifier.tri)
    
def plot_classifier_scrollable(Classifier: Classifier,it,long_data,long_labels,long_tris,err_dict,run_params):
    plot_scrollable_data(Classifier.data,Classifier.labels,Classifier.sample,Classifier.rem,Classifier.dim,Classifier.dim_labels,it,long_data,long_labels,long_tris,err_dict,run_params)

def plot_metrics(Measurer: Measurer):
    fig, ax = plt.subplots(2,2)
    metrics = ['training_error','error_variance','edge_variance','real_error']
    for i in range(len(metrics)):
        metric = metrics[i]
        ax[i//2,i%2].plot(range(len(Measurer.metrics[metric])),Measurer.metrics[metric],'--+')
        ax[i//2,i%2].set_title(metric)
    
    return fig