# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from functions.reading_functions import extract_run_params
from functions.class_functions import fully_train
from functions.plotting_functions import plot_classifier, plot_classifier_scrollable, plot_metrics
from classes.Classifier import *
from classes.Measurer import Measurer
from classes.Reader import Reader


folder_name = 'diabetes_pp'
Reader = Reader(folder_name)
Classifier = Reader.read_classifier()
Measurer = Reader.read_measurer()
fig_classifier = plot_classifier(Classifier)
fig_metrics = plot_metrics(Measurer) 
plt.show()