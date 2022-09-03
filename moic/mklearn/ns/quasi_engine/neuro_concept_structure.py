import torch
import torch.nn as nn

from aluneth.rinlearn.ns.quasi_engine.neuro_types import *
# A concept structure is convieved as the way how concepts/relations interacts with object/objects
# and produce the probabilitic outcomes. Basically a concept structure is a set of functions that 
# contains the way to calculate the following things:
# Pr[c|e] which means the outcome prob of e entails c 
# Pr[r|e1,e2] which means the prob of (e1,e1) entails r


# Basic concept types of for the concept structure

class ConceptDot(nn.Module):
    def __init__(self,name,type,dim = 256):
        super().__init__()
        self.name = name
        self.type = type
        self.feature = nn.Parameter(torch.randn([1,dim]))

    def __str__(self):return "concept:{}".format(self.name)

class RelationDot(nn.Module):
    def __init__(self,name,type,dim = 512):
        super().__init__()
        self.name = name
        self.type = type
        self.feature = nn.Parameter(torch.randn([1,dim]))

    def __str__(self):return "relation:{}".format(self.name)
# This part contains the basic concept type and corresponding structures.

class NeuroConceptStructure(nn.Module):
    def __init__(self,concepts,relations):
        super().__init__()
        if not isinstance(concepts,nn.ModuleList):
            self.concepts = nn.ModuleList(concepts)
        else:self.concepts = concepts

        if not isinstance(relations,nn.ModuleList):
            self.relations = nn.ModuleList(relations)
        else:self.relations = relations

        self.concept_links = {}
        for c in concepts:
            if c.type in self.concept_links.keys():
                self.concept_links[c.type].append(c.name)
            else:
                self.concept_links[c.type] = [c.name]

        self.relation_links = {}
        for r in relations:
            if r.type in self.relation_links.keys():
                self.relation_links[r.type].append(r.name)
            else:
                self.relation_links[r.type] = [r.name]    

    def add_concept(self,concept): self.concepts.append(concept)

    def MeasureConcept(self,concept,entity): 
        measures = []
        assert torch.abs(torch.sum(entity.probs)-1)<0.1,print("Not a valid single object")
        for i in range(entity.features.shape[0]):
            e = entity.features[i]
            scores = []
            concept_values = []
            for c in self.concepts:
                if c.type == concept:
                    concept_values.append(c.name)
                    scores.append(torch.sigmoid( (torch.cosine_similarity(c.feature,e)-0.2) / 0.15))
            scores = normalize(torch.cat(scores,0))
 
            meas = ConceptMeasurement(concept_values,scores)
            measures.append(meas)

        return mix_measurements(measures,entity.probs)

            
    def PrConceptMeasure(self,value,entity):
        scores = [];concept_values = []
        father_type = None
        for c in self.concepts:
            if c.name == value and father_type == None:
                father_type = c.type
                break # evaluate the type of the concept
        assert father_type != None,print("Father type of {} not found: {}".format(value,father_type))
        e = entity
        for c in self.concepts:
            if c.type == father_type:
                concept_values.append(c.name)
                scores.append(torch.sigmoid( (torch.cosine_similarity(c.feature,e)-0.2) / 0.15))
        assert len(scores)>0,print("There is no concept under the category of {}".format(father_type))
        scores = normalize(torch.cat(scores,0))
 
        return scores[concept_values.index(value)]
        

    def MeasureRelation(self,relation,entity1,entity2):
        measures = [];mix_probs = []
        assert torch.abs(torch.sum(entity1.probs)-1)<0.1,print("Not a valid single object 1")
        assert torch.abs(torch.sum(entity2.probs)-1)<0.1,print("Not a valid single object 2")
        for i in range(entity1.features.shape[0]):
            for j in range(entity2.features.shape[0]):
                e1 = entity1.features[i]
                e2 = entity2.features[j]
                mix_probs.append(entity1.probs[i] * entity2.probs[j])
                scores = []
                relation_values = []
                for r in self.relations:
                    if r.type == relation:
                        relation_values.append(r.name)
                        e = torch.cat([e1,e2],0)
                        scores.append(torch.sigmoid( (torch.cosine_similarity(r.feature,e)-0.2) / 0.15))
                scores = normalize(torch.cat(scores,0))
                meas = ConceptMeasurement(relation_values,scores)
                measures.append(meas)
        return mix_measurements(measures,mix_probs)

    def PrRelationMeasure(self,specific_relation,entity1,entity2):
        father_type = None
        for r in self.relations:
            if r.name == specific_relation and father_type == None:
                father_type = r.type
                break # evaluate the type of the concept
        assert father_type != None,print("Father type of {} not found: {}".format(specific_relation,father_type))
        se1 = SingleObject(entity1,torch.ones([1]))
        se2 = SingleObject(entity2,torch.ones([1]))
        measure_pdf = self.MeasureRelation(father_type,se1,se2)
        return measure_pdf.probs[measure_pdf.keys.index(specific_relation)]

