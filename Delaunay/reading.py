# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from functions.reading_functions import extract_run_params
from functions.class_functions import fully_train
from functions.plotting_functions import plot_classifier, plot_classifier_scrollable, plot_metrics
from classes.Classifier import *
from classes.Trainer import Trainer
from classes.Measurer import Measurer
from classes.Writer import Writer
from classes.Reader import Reader

data_name = 'yeast'
trainers_prefixes = ['balanced_gradient','error_gradient','distance_gradient']
trainers_sufixes = ['simple','barycenters','refine','full']
metrics = ['training_error','error_variance','edge_variance','real_error']
def add_plot(ax,data_name,prefix,sufix,metric):
    from classes.Reader import Reader
    folder_name = data_name+'_'+prefix+'_'+sufix
    Reader = Reader(folder_name)
    Measurer = Reader.read_measurer()
    ax.plot(range(len(Measurer.metrics[metric])),Measurer.metrics[metric],'--+',label=sufix)
    ax.set_title(metric+'_'+prefix)
    ax.legend()

fig, ax = plt.subplots(4,3, figsize=(15,10))

for i in range(len(metrics)):
    metric = metrics[i]
    for j in range(len(trainers_prefixes)):
        prefix = trainers_prefixes[j]
        for sufix in trainers_sufixes:
            add_plot(ax[i,j],data_name,prefix,sufix,metric)

""" fig, ax = plt.subplots()

prefix = 'error_gradient'
for sufix in trainers_sufixes:
    add_plot(fig,ax,data_name,prefix,sufix) """

""" 
Reader = Reader(folder_name)
Classifier = Reader.read_classifier()
Measurer = Reader.read_measurer()
fig_classifier = plot_classifier(Classifier)
fig_metrics = plot_metrics(Measurer) 
"""
plt.show()