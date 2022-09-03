
import torch
import torch.nn as nn

import numpy as np

from aluneth.utils import *

class Rint(nn.Module):
    def __init__(self,value):
        super().__init__()
        self.value = value
    
    def pdf(self,flag = False):return int(dnp(self.value) + 0.5),self.value

class ObjectSet(nn.Module):
    def __init__(self,features,probs):
        super().__init__()
        assert probs.shape[0] == features.shape[0],"size of features and probs don't match"
        self.features = nn.Parameter(features)
        self.probs = nn.Parameter(probs)
    def object_set(self): return 0
    def pdf(self,flag= False): return dnp(self.probs)
    
class SingleObject(nn.Module):
    def __init__(self,features,probs,cast=True):
        super().__init__()
        assert probs.shape[0] == features.shape[0],"size of features and probs don't match"
        self.features = nn.Parameter(features)
        self.probs = nn.Parameter(probs)
    def pdf(self): return dnp(self.probs)

def normalize(tensor): return tensor/torch.sum(tensor)

def cast_object_set(OSet):return SingleObject(OSet.features,normalize(OSet.probs))


class ConceptMeasurement(nn.Module):
    def __init__(self,keys,probs,cast = True):
        super().__init__()
        assert probs.shape[0] == len(keys),"size of keys and probs don't match"
        self.keys = keys
        self.probs = probs
        if cast: self.probs = self.probs/torch.sum(self.probs)
    
    def pdf(self,flag=False): 
        if flag:return self.keys,dnp(self.probs)
        else: return self.keys,self.probs
    
    def most_likely_result(self):return self.keys[np.argmax(dnp(self.probs))]

    def sample_result(self):return np.random.choice(self.keys,p=dnp(self.probs))

def mix_measurements(measurements,probs,unitary = False):
    size_meas = 0;size_probs = -1
    if isinstance(measurements,torch.Tensor):size_meas = measurements.shape[0] 
    else: size_meas = len(measurements)
    if isinstance(probs,torch.Tensor):size_probs = probs.shape[0] 
    else: size_probs = len(probs)
   
    assert size_meas  == size_probs,"measurements and mix probs doesn't match"
    if isinstance(probs,list):probs = torch.tensor(probs)
    if(unitary): probs = probs/torch.sum(probs)

    basic_meas = measurements[0].probs * probs[0]
    for i in range(len(measurements)-1):
        meas = measurements[i+1]
        basic_meas = basic_meas + meas.probs * probs[i + 1]
    return ConceptMeasurement(measurements[0].keys,basic_meas)
