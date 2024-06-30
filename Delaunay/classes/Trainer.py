from classes.Classifier import Classifier

from functions.training_functions import delaunayization, train_step, add_barycenters

class Trainer:
    def __init__(self,Classifier: Classifier, run_params):
        self.al = run_params['al']
        self.errw = run_params['errw']
        self.avw = run_params['avw']
        self.bc_time = run_params['bc_time']
        self.mte_threshold = run_params['mte_threshold']
        e, err = delaunayization(Classifier,self.avw)
        self.e = e
        self.err = err

    def first_train(self,Classifier: Classifier):
        self.e, self.err = delaunayization(Classifier,self.avw)
        

    def train(self,Classifier: Classifier):
        train_step(Classifier,self.al,self.errw,self.err)
        self.e, self.err = delaunayization(Classifier,self.avw)
        
    def add_barycenters(self,Classifier: Classifier):
        added = add_barycenters(Classifier, self.e)
        return added
    