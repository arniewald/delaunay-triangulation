import numpy as np
import pandas as pd
import random
import json

from sklearn.datasets import load_iris

from functions import *
from mc_functions import *


def read_iris_data(features = None, errw = 0.5, ref_label = None, bc_time = np.inf, mte_threshold = 0.5, seed = 0):
    """
    For 3D: features = ['sepal length (cm)','petal length (cm)', 'petal width (cm)']
    """
    if features == None:
        iris = load_iris()
        data = iris['data']
        labels = np.array(iris['target'], dtype=float)
    else:
        iris = load_iris(as_frame=True)['frame'] #We discard cepal width
        iris = iris[features+['target']] #There is repeated data since we are removing a feature, so we remove duplicates
        iris.drop_duplicates(inplace=True)
        data = iris[features].to_numpy()
        data = np.array(data, dtype=float)
        labels = iris['target'].to_numpy()
        labels = np.array(labels, dtype=float)


    binary = ''
    if ref_label!=None:
        binary = '_binary'+str(ref_label)
        for i in range(len(labels)):
            if labels[i] == ref_label:
                labels[i] = 1
            else:
                labels[i] = 0
    #Name of file in which to write results
    filename = '_Iris4D_mc' + binary + '_' + str(errw) + '_bctime' + str(bc_time) + '_th' + str(mte_threshold)

    return data, labels, filename

def read_beans_data(classes = ['BARBUNYA','BOMBAY','CALI'], features = ['Area', 'Perimeter', 'MajorAxisLength','MinorAxisLength','AspectRation'], errw = 0.5, ref_label = None, bc_time = np.inf, mte_threshold = 0.5, seed = 0):
    """ 
    Classes : 'BARBUNYA','BOMBAY','CALI','DERMASON','HOROZ','SEKER','SIRA'
    Fields : 'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
        'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent',
        'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
        'ShapeFactor3', 'ShapeFactor4', 'Class' 
    """
    random.seed(seed)
    class_map = dict()
    for i in range(len(classes)):
        class_map[classes[i]]=i
    df = pd.read_csv('data/Dry_Bean_Dataset.csv')
    df = df[df.Class.isin(classes)]
    df = df.replace({'Class':class_map})
    if features!=None:
        df = df[features+['Class']]
        df.drop_duplicates(inplace=True)
        data = np.array(df[features].to_numpy(), dtype = np.float64)
    else:
        data = np.array(df[list(df.columns)[:-1]].to_numpy(), dtype = np.float64)
    labels = np.array(df['Class'].to_numpy(), dtype = np.float64)

    dim = data.shape[1]  #Dimension of data
    size = math.floor(len(data)*0.05)               #Original size of Delaunay triangulation
    test_size = math.floor(len(data)*0.1)          #Size of subdata to test

    str_features = ''
    if features!=None:
        str_features = str_features.join(features)
        str_features = str_features + '_'
    filename = 'Beans_errorgradient_'+str(errw) + str_features + str(ref_label) + '_bctime' + str(bc_time) + '_th' + str(mte_threshold)
    return data, labels, dim, size, test_size, filename

def read_yeast_data(classes = ['NUC','ME1','ME2'], features = ['mcg','gvh','mit','erl'], errw = 0.5, ref_label = None, bc_time = np.inf, mte_threshold = 0.5, seed = 0):
    """
    Classes : 'CYT','ERL','EXC','ME1','ME2','ME3','MIT','NUC','POX','VAC'
    Fields : 'Sequence_Name','mcg','gvh','alm','mit','erl','pox','vac','nuc','localization_site'
    """
    
    random.seed(seed)
    class_map = dict()
    for i in range(len(classes)):
        class_map[classes[i]]=i
    #Extraction of data with the desired features and classes
    df = pd.read_csv('data/yeast.csv')
    df = df[df['localization_site'].isin(classes)]
    df = df.replace({'localization_site':class_map})
    if features!=None:
        df = df[features+['localization_site']]
        df.drop_duplicates(inplace=True)
        data = np.array(df[features].to_numpy(), dtype = np.float64)
    else:
        data = np.array(df[list(df.columns)[:-1]].to_numpy(), dtype = np.float64)
    labels = np.array(df['localization_site'].to_numpy(), dtype = np.float64)
    dim = data.shape[1]  #Dimension of data
    size = math.floor(len(data)*0.3)               #Original size of Delaunay triangulation
    test_size = math.floor(len(data)*0.1)          #Size of subdata to test

    str_features = ''
    if features!=None:
        str_features = str_features.join(features)
        str_features = str_features + '_'
    filename = 'Yeast_NUC_CYT_MIT_errorgradient_'+str(errw) + str_features + str(ref_label) + '_bctime' + str(bc_time) + '_th' + str(mte_threshold)
    return data, labels, dim, size, test_size, filename

def read_general_data(data_name):
    f = open('params.json')
    args = json.load(f)[data_name]
    if data_name == 'iris':
        return read_iris_data(**args)
    elif data_name == 'beans':
        return read_beans_data(**args)
    elif data_name == 'yeast':
        return read_yeast_data(**args)
    else:
        print('Error: incorrect name of dataset')
        return None