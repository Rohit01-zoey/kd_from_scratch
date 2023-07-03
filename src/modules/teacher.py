'''Implementation of the teacher model in the KD set up'''
import torch.functional as F
from config import cfg
from model import *

class Teacher():
    '''The Teacher class module for the KD set up'''
    def __init__(self, model, dataset):
        self.model = model # instantiate the teacher class with the model definition
        self.dataset = dataset # instantiate the teacher class with the dataset definition
        
    def label_dataset(self, dataset):
        '''Label the dataset using the teacher model'''
        self.model.eval()
        with torch.no_grad(): # set the model to inference mode
            for k in dataset.keys():
                dataset[k]["soft_targets"] = F.softmax(self.model(dataset[k]["data"]), dim = -1) # compute the soft labels for the dataset
        return dataset # return the soft labelled dataset for the student model
    
    def save_model(self, path):
        '''Save the model'''
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        '''Load the model'''
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        return self.model
