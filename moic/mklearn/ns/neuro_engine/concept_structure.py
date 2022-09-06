import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from abc import abstractmethod

def softplus(x):
    t = 0.2
    #return torch.max(x,torch.zeros_like(x))
    return t * torch.log(1 + torch.exp(x/t))

def bound(x,length = 0.4):
    return torch.tanh(x) * length


class ConceptBox(nn.Module):
    def __init__(self,token,s_dim = 100):
        super().__init__()
        self.token = token
        self.center = nn.Parameter(torch.randn([1,s_dim]) )
        self.edge = nn.Parameter(torch.randn([1,s_dim]))
        self.d = 0.25
        self.structure = None
        self.s_dim = s_dim
    def Center(self):
        d = self.d
        return d * torch.tanh(self.center)

    def Edge(self):
        d = self.d
        return d * torch.sigmoid(self.edge)

    def Min(self):
        d = self.d
        return self.Center() - self.Edge()

    def Max(self):
        d = self.d
        return self.Center() + self.Edge()

class EntityBox(nn.Module):
    def __init__(self,feature,s_dim = 100,entity_edge_length = 1e-4):
        super().__init__()
        self.token = "entity"
        self.s_dim = feature.shape[1]
        self.center = feature
        self.edge = entity_edge_length * torch.ones([1,self.s_dim])
        self.entity_edge_length = entity_edge_length
        self.d = 0.25

    def Center(self):
        d = self.d
        return d * torch.tanh(self.center)

    def Edge(self):
        d = self.entity_edge_length
        return self.edge

    def Min(self):
        d = self.d
        return self.Center() - self.Edge()

    def Max(self):
        d = self.d
        return self.Center() + self.Edge()

class ConceptDot(nn.Module):
    def __init__(self,name,dim = 256):
        super().__init__()
        self.s_dim = dim
        self.center = nn.Parameter(torch.randn([1,dim]))
        self.token = name
    
    def Semantics(self):
        return self.center

class EntityDot(nn.Module):
    def __init__(self,reps,dim):
        super().__init__()
        self.s_dim = reps.shape[0]
        self.center= reps
    
    def Semantics(self):
        return self.center

def M(concept1,concept2):
    return torch.min(concept1.Max(),concept2.Max())

def m(concept1,concept2):
    return torch.max(concept1.Min(),concept2.Min())

def ProbC(concept):
    output = torch.prod(concept.Max() - concept.Min())
    return output

def LogProbC(concept):

    output = torch.log(concept.Max() - concept.Min())

    output = torch.sum(output)

    return output

def JointProb(concept1,concept2,flag = False):
    if (not flag):

        return torch.prod(softplus(M(concept1,concept2) - m(concept1,concept2)))
    else:
        try:
            return torch.prod(torch.max(M(concept1,concept2) - m(concept1,concept2),torch.zeros([concept1.s_dim,1])))
        except:
            return torch.prod(torch.max(M(concept1,concept2) - m(concept1,concept2),torch.zeros(concept2)))
def LogJointProb(concept1,concept2):
    #print("actual",torch.log(torch.prod(softplus(M(concept1,concept2) - m(concept1,concept2)))))
    #return torch.log(torch.prod())
    return torch.sum(torch.log(softplus(M(concept1,concept2) - m(concept1,concept2))))

def ConditionProb(c1,c2,flag = False):
    if (flag):
        return JointProb(c1,c2,True)/ProbC(c2)
    return JointProb(c1,c2)/ProbC(c2)

def LogConditionProb(c1,c2):
    return LogJointProb(c1,c2) - LogProbC(c2)

def draw_boxes(concepts):
    plt.ylim(ymin = -1/2,ymax = 1/2)
    plt.xlim(xmin = -1/2,xmax = 1/2)
    for c in concepts:
        ct = c.Center()[0].detach().numpy()
        eg = c.Edge()[0].detach().numpy()
        a,b = c.Edge()[0][0].detach().numpy(), c.Edge()[0][1].detach().numpy()
        plt.text(ct[0]-eg[0],ct[1]-eg[1],c.token)
        plt.gca().add_patch(
            Rectangle((c.Center()-c.Edge())[0].detach().numpy(),2*a,2*b ,edgecolor='red',facecolor='none')
            )
    return 0

class ConceptStructure(nn.Module):
    def __init__(self,constants):
        super().__init__()
    
    @abstractmethod
    def ObjClassify(self,entity,concept):
        return 0
    
    @abstractmethod
    def Classify(self,entity,concept):
        return 0
    
    @abstractmethod
    def Relate(self,entity,concept):
        return 0

def toBoxConcepts(constants,dim = 100):
    output = nn.ModuleList([])
    for const in constants:
        output.append(ConceptBox(const.token,dim))
    return output

def toDotConcepts(constants,dim = 256):
    output = nn.ModuleList([])
    for const in constants:
        output.append(ConceptDot(const.token,dim))
    return output

class BoxConceptStructure(ConceptStructure):
    def __init__(self,constants):
        super().__init__(constants)
        self.concept_diction = {}
        self.concept_names = []
        for const in constants:
            if (const.token not in self.concept_names):
                self.concept_names.append(const.token)
            if (const.out_type  not in self.concept_names):
                self.concept_names.append(const.out_type)
            ctype = const.out_type
            if (ctype in self.concept_diction.keys()):
                self.concept_diction[ctype].append(const)
            else:
                self.concept_diction[ctype] = [const]
        #print("concept structure diciton:",self.concept_diction)
        self.dim = constants[0].s_dim
        #print("structure dim:",dim)
        self.constants = toBoxConcepts(constants,self.dim)
    
    def Posses(self,entities,concept):
        # input a set of box entities and key-level concept
        return 0
    
    def Classify(self,entity,concept):
        # input a entity and a category level concept.
        # output the keys and probs for the enitity belongs to each key
        constant_key = None
        for const in self.constants:

            if const.token == concept:
                constant_key = const

        if (constant_key == None):
            print("Error: concept key for classification not found")
            return -1
        #print("Pr[e|c]",ConditionProb(EntityBox(entity,constant_key.s_dim),constant_key,True))
        #print("Pr[c|e]",ConditionProb(constant_key,EntityBox(entity,constant_key.s_dim),True))
        #print("LogPr[e|c]", LogConditionProb(EntityBox(entity,constant_key.s_dim),constant_key,True))
        return LogConditionProb(EntityBox(entity,constant_key.s_dim),constant_key)

    def ObjClassify(self,entity,concept):
        # return Pr[o|c] 
        constant_key = None
        for const in self.constants:
            if const.token == concept:
                constant_key = const
        if (constant_key == None):
            print("Error: concept key for classification not found")
            return -1

        return LogConditionProb(entity,constant_key)
    
    def getFatherKey(self,concept):
        for key in self.concept_diction.keys():
            values = self.concept_diction[key]
            for val in values:
                if (val.token == concept):
                    return key
        return -1
    
    def getConcept(self,concept):
        for cons in self.constants:
            if (cons.token == concept):
                return cons
        return -1

    def FilterClassify(self,entity,concept):

        cateName = self.getFatherKey(concept)
        values = self.concept_diction[cateName]
        target = self.getConcept(concept)

        Prb = torch.exp(LogConditionProb(entity,target))
        Deno = 0
        for val in values:
            Deno = Deno + torch.exp(self.ObjClassify(entity,val.token))
        return Prb/Deno

    def Relate(self,entity,concept):
        return 0

class ConeConceptStructure(nn.Module):
    def __init__(self,constants,dim_c = 100, dim_object = 256):
        super().__init__()
        self.concept_diction = {}
        self.concept_names = []
        for const in constants:
            if (const.token not in self.concept_names):
                self.concept_names.append(const.token)
            if (const.out_type  not in self.concept_names):
                self.concept_names.append(const.out_type)
            ctype = const.out_type
            if (ctype in self.concept_diction.keys()):
                self.concept_diction[ctype].append(const)
            else:
                self.concept_diction[ctype] = [const]
        #print("concept structure diciton:",self.concept_diction)
        self.dim = constants[0].s_dim
        #print("structure dim:",dim)
        self.constants = toDotConcepts(constants,self.dim)

    def Relate(self):
        return 0
    
    def Classify(self,entity,concept):
        # input a entity and a category level concept.
        # output the keys and probs for the enitity belongs to each key
        constant_key = None
        for const in self.constants:

            if const.token == concept:
                constant_key = const

        if (constant_key == None):
            print("Error: concept key for classification not found")
            return -1
        #print("Pr[e|c]",ConditionProb(EntityBox(entity,constant_key.s_dim),constant_key,True))
        #print("Pr[c|e]",ConditionProb(constant_key,EntityBox(entity,constant_key.s_dim),True))
        #print("LogPr[e|c]", LogConditionProb(EntityBox(entity,constant_key.s_dim),constant_key,True))
        return LogConditionProb(EntityBox(entity,constant_key.s_dim),constant_key)

    def ObjClassify(self,entity,concept):
        # return Pr[o|c] 
        constant_key = None
        for const in self.constants:
            if const.token == concept:
                constant_key = const
        if (constant_key == None):
            print("Error: concept key for classification not found")
            return -1
        gamma = 0.2
        tau = 0.1
        return torch.log(torch.sigmoid( (torch.cosine_similarity(entity.Semantics(),constant_key.Semantics()) - gamma )/ tau ))

    def getFatherKey(self,concept):
        for key in self.concept_diction.keys():
            values = self.concept_diction[key]
            for val in values:
                if (val.token == concept):
                    return key
        return -1
    
    def getConcept(self,concept):
        for cons in self.constants:
            if (cons.token == concept):
                return cons
        return -1

    def FilterClassify(self,entity,concept):

        cateName = self.getFatherKey(concept)
        values = self.concept_diction[cateName]
        target = self.getConcept(concept)

        Prb = torch.exp(self.ObjClassify(entity,target.token))
        
        Deno = 0
        for val in values:
            Deno = Deno + torch.exp(self.ObjClassify(entity,val.token))
    
        return Prb/Deno

print("Quasi-Symbolic Concept Structure Loaded.")