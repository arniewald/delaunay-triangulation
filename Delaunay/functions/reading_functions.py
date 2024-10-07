import numpy as np
import pandas as pd
import random
import json

def extract_data_params(data_name):
    """
    Extracts parameters of a dataset. 
    The type of parameters are different for each dataset.
    For synthetic datasets, it usually contains the size, possible noise and dimension.
    For real datasets, it usually selects some features and some possible target labels.
    
    Args:
        - data_name : name of the dataset.

    Returns:
        - data_params : dictionary containing the parameters of the dataset.
    """
    f = open('jsons\\data_params.json')
    data_params = json.load(f)[data_name]
    return data_params

def extract_run_params(data_name):
    """
    Extracts the parameters that characterize how a classifier will be initialized and trained.
    The structure is:
        - size_prop : proportion of the original dataset that will be used as classification points (points to construct the Delaunay triangulation).
        - sampling : type of sampling to draw the classification points, either "random" or "entropic".
        - test_size : number of points to be used for computing the real error.
        - al : measures the magnitude of the overall displacement.
        - errw :  weight of the error gradient.
        - avw :  weight of the averages equations.
        - rep : number of times to refine the convex hull.
        - bc_time : time at which to add barycenters.
        - mte_threshold : threshold above which the barycenters will be added.
        - seed : seed used when drawing the classification points.
    """
    f = open('jsons\\run_params.json')
    run_params = json.load(f)[data_name]
    return run_params

def read_circles_data(n_samples = 10000, noise = 0.1):
    """
    Reads the circles dataset.

    Args:
        - n_samples : number of points.
        - noise : noise added to the dataset, the bigger the more mixed.

    Returns:
        Tuple containing
            - data : data from which the classifier will be created.
            - labels : labels of data.
            - dim : dimension of data.
    """
    from sklearn.datasets import make_circles
    data, labels = make_circles(n_samples=n_samples, noise=noise)
    data = np.array(data, dtype=float)
    labels = np.array(labels, dtype=float)
    dim = 2
    return data, labels, dim

def read_moons_data(n_samples = 10000, noise = 0.1):
    """
    Reads the moons dataset.

    Args:
        - n_samples : number of points.
        - noise : noise added to the dataset, the bigger the more mixed.

    Returns:
        Tuple containing
            - data : data from which the classifier will be created.
            - labels : labels of data.
            - dim : dimension of data.
    """
    from sklearn.datasets import make_moons
    data, labels = make_moons(n_samples = n_samples, noise = noise)
    data = np.array(data, dtype=float)
    labels = np.array(labels, dtype=float)
    dim = 2
    return data, labels, dim

def read_classification_data(n_samples,n_features,n_classes):
    """
    Reads the scikit classification dataset.

    Args:
        - n_samples : number of points.
        - n_features : number of features of data.
        - n_classes : number of possible labels.

    Returns:
        Tuple containing
            - data : data from which the classifier will be created.
            - labels : labels of data.
            - dim : dimension of data.
    """
    from sklearn.datasets import make_classification
    data, labels = make_classification(n_samples=n_samples,n_features=n_features, n_redundant=0, n_informative=2, n_clusters_per_class=1,n_classes=n_classes,random_state=0)
    data = np.array(data, dtype=float)
    labels = np.array(labels, dtype=float)
    dim = n_features
    return data, labels, dim

def read_iris_data(features = None, ref_label = None):
    """
    Reads the iris dataset.

    Args:
        - features : features to select of data, all by default.
        - ref_label : if not None, binarizes the labels. 

    Returns:
        Tuple containing
            - data : data from which the classifier will be created.
            - labels : labels of data.
            - dim : dimension of data.
    """
    #For 3D: features = ['sepal length (cm)','petal length (cm)', 'petal width (cm)']
    from sklearn.datasets import load_iris
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

    dim = data.shape[1]
    return data, labels, dim

def read_beans_data(classes = ['BARBUNYA','BOMBAY','CALI'], features = ['Area', 'Perimeter', 'MajorAxisLength','MinorAxisLength','AspectRation']):
    """ 
    Reads the beans data.

    Classes : 'BARBUNYA','BOMBAY','CALI','DERMASON','HOROZ','SEKER','SIRA'
    Fields : 'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
        'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent',
        'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
        'ShapeFactor3', 'ShapeFactor4', 'Class' 

    Args:
        - classes : possible labels to select.
        - features : features of data.
    
    Returns:
        Tuple containing
            - data : data from which the classifier will be created.
            - labels : labels of data.
            - dim : dimension of data.
    """
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

    return data, labels, dim

def read_yeast_data(classes = ['NUC','ME1','ME2'], features = ['mcg','gvh','mit','erl']):
    """ 
    Reads the yeast data.

    Classes : 'CYT','ERL','EXC','ME1','ME2','ME3','MIT','NUC','POX','VAC'
    Fields : 'Sequence_Name','mcg','gvh','alm','mit','erl','pox','vac','nuc','localization_site'
    
    Args:
        - classes : possible labels to select.
        - features : features of data.
    
    Returns:
        Tuple containing
            - data : data from which the classifier will be created.
            - labels : labels of data.
            - dim : dimension of data.
    """
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
    return data, labels, dim

def read_franco_data(dim,holes = [[-2,3,3],[-4,-4,4]],n_samples=5000,noise=0.3):
    """
    Generates and reads franco dataset.

    Args:
        - dim :  dimension of data.
        - holes : coordinates of the holes present in the data.
        - n_samples : number of points.
        - noise : noise added to the dataset, the bigger the more mixed.

    Returns:
        Tuple containing
            - data : data from which the classifier will be created.
            - labels : labels of data.
            - dim : dimension of data.
    """
    data = np.zeros((n_samples,dim))
    for i in range(dim):
        data[:,i] = -2+4*np.array([np.random.normal() for _ in range(n_samples)])
    for hole in holes:
        data = np.array([x for x in data if sum((x-np.array(hole[:-1]))**2)>hole[-1]])
    labels = np.array([int((x[0]**2 + x[1]**2)**2 - 2*49*(x[0]**2 - x[1]**2) > 0) for x in data])
    data += noise*np.array([np.random.normal(size=[1,dim])[0] for _ in range(len(data))])
    return data, labels, dim

def read_spheres_data(dim,n_spheres,n_samples_per_sphere):
    """
    Generates and reads the spheres dataset.

    Args:
        - dim : dimension of data.
        - n_spheres : number of spheres that the data form.
        - n_samples_per_sphere : number of points for each sphere.

    Returns:
        Tuple containing
            - data : data from which the classifier will be created.
            - labels : labels of data.
            - dim : dimension of data.
    """
    data = np.zeros((n_spheres*n_samples_per_sphere,dim))
    labels = np.zeros(n_spheres*n_samples_per_sphere)
    for i in range(n_spheres):
        orientation = [np.cos(np.pi/2*i),np.sin(np.pi/2*i),0]
        for j in range(dim):
            data[n_samples_per_sphere*i:n_samples_per_sphere*(i+1),j] = -5*orientation[j]+4*np.array([np.random.normal() for _ in range(n_samples_per_sphere)])
            labels[n_samples_per_sphere*i:n_samples_per_sphere*(i+1)] = i
    return data, labels, dim

def read_anemia_data(features = ['%Red Pixel','%Green pixel','%Blue pixel']):
    """
    Generates and reads the anemia dataset.

    Args:
        - features : features to select from the dataset.

    Returns:
        Tuple containing
            - data : data from which the classifier will be created.
            - labels : labels of data.
            - dim : dimension of data.
    """
    #Extraction of data with the desired features and classes
    df = pd.read_csv('data/anemia.csv')
    df.Anaemic = np.where(df.Anaemic=='Yes',1,0)
    if features!=None:
        df = df[features+['Anaemic']]
        df.drop_duplicates(inplace=True)
        data = np.array(df[features].to_numpy(), dtype = np.float64)
    else:
        df = df.drop(['Number','Sex'],axis=1)
        data = np.array(df[list(df.columns)[:-1]].to_numpy(), dtype = np.float64)
    labels = np.array(df['Anaemic'].to_numpy(), dtype = np.float64)
    dim = data.shape[1]  #Dimension of data
    return data, labels, dim

def read_diabetes_data(features = ['Glucose','BloodPressure','Insulin']):
    """
    Generates and reads the diabetes dataset.

    Args:
        - features : features to select from the dataset.

    Returns:
        Tuple containing
            - data : data from which the classifier will be created.
            - labels : labels of data.
            - dim : dimension of data.
    """
    #Extraction of data with the desired features and classes
    df = pd.read_csv('data/diabetes.csv')
    if features!=None:
        df = df[features+['Outcome']]
        df.drop_duplicates(inplace=True)
        data = np.array(df[features].to_numpy(), dtype = np.float64)
    else:
        data = np.array(df[list(df.columns)[:-1]].to_numpy(), dtype = np.float64)
    labels = np.array(df['Outcome'].to_numpy(), dtype = np.float64)
    dim = data.shape[1]  #Dimension of data
    return data, labels, dim

reading_functions_dict = {
    'circles': read_circles_data,
    'moons': read_moons_data,
    'classification': read_classification_data,
    'iris': read_iris_data,
    'beans': read_beans_data,
    'yeast': read_yeast_data,
    'franco': read_franco_data,
    'spheres': read_spheres_data,
    'anemia': read_anemia_data,
    'diabetes': read_diabetes_data
}

def read_general_data(data_name):
    """
    Reads one of the datasets available and extracts its parameters.

    Args:
        - data_name : dataset to be read.

    Returns:
        Tuple containing
            - data_params : dictionary containing the parameters of the dataset.
            - data : data from which the classifier will be created.
            - labels : labels of data.
            - dim : dimension of data.
    """
    data_params = extract_data_params(data_name)
    try:
        data, labels, dim = reading_functions_dict[data_name](**data_params)
        return data_params, data, labels, dim
    except KeyError as e:
        print(e)