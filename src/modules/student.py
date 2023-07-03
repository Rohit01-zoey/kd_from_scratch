'''Implementation of the Student model in the KD set up'''
import torch.functional as F
from config import cfg
from model import *


class Student():
    '''The Student class module for the KD set up'''
    def __init__(self, model, dataset):
        self.model = model # instantiate the teacher class with the model definition
        self.dataset = dataset # instantiate the teacher class with the dataset definition
        
    def save_model(self, path):
        '''Save the model'''
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        '''Load the model'''
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        return self.model