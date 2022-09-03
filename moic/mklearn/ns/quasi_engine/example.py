from neuro_config import *
from neuro_vertex import *
from neuro_concept_structure import *
from neuro_types import *

# start the session of declareing concepts and relations
c1 = ConceptDot("blue","color")
c2 = ConceptDot("red","color")
c3 = ConceptDot("green","color")
# color concept of red blue green

r1 = RelationDot("left","position")
r2 = RelationDot("right","position")
# positional relation of left and right

clist = nn.ModuleList([c1,c2,c3])
rlist = nn.ModuleList([r1,r2])

for c in clist:print(c)
for r in rlist:print(r)
# display all the relations and concepts

# create the concept structure that calculate the concepts
cstructure = NeuroConceptStructure(clist,rlist)

# write some basic implementations of these functions

class CScene(DiffVertex):
    def __init__(self):
        super().__init__()
        self.name = "scene"

    def prop(self,inputs,structure,context):
        assert isinstance(context,dict),print("context is not a valid diction.")
        return context["Objects"]

class CMeasureColor(DiffVertex):
    def __init__(self):
        super().__init__()
        self.name = "measure_color"

    def prop(self,inputs,structure,context):
        assert isinstance(context,dict),print("context is not a valid diction.")

        return structure.MeasureConcept("color",inputs[0])

class CMeasureRelation(DiffVertex):
    def __init__(self):
        super().__init__()
        self.name = "measure_relation"
    
    def prop(self,inputs,structure,context):
        assert isinstance(context,dict),print("context is not a valid diction.")
        return structure.MeasureRelation("position",inputs[0],inputs[1])

class CFilterColor(DiffVertex):
    def __init__(self):
        super().__init__()
        self.name = "filter_color"
    
    def prop(self,inputs,structure,context):
        input_color = 'blue'
        input_set = inputs[1]
        filter_scores = []
        for i in range(input_set.features.shape[0]):
            prior_prob = input_set.probs[i]
            filter_prob = structure.PrConceptMeasure(input_color,input_set.features[i])
            filter_scores.append(torch.min(prior_prob,filter_prob).reshape([1,-1]))

        return ObjectSet(input_set.features,torch.cat(filter_scores))
class CFilterRelation(DiffVertex):
    def __init__(self):
        super().__init__()
        self.name = "filter_relation"
    def prop(self,inputs,structure,context):
        input_object = inputs[0]
        input_sets = inputs[1]
        input_relation = inputs[2]
        #input_relation = "left"
        output_scores = []
        for i in range(input_sets.features.shape[0]):
            e1 = input_object.features[i:i+1]
            e2 = input_sets.features[i:i+1]
            score = structure.PrRelationMeasure(input_relation,e1,e2) * input_object.probs[i] * input_sets.probs[i]
            output_scores.append(torch.min(score,input_sets.probs[i]).reshape([1,1]))
        output_scores = torch.cat(output_scores,0)
        return ObjectSet(input_sets.features,output_scores)

class CUnique(DiffVertex):
    def __init__(self):
        super().__init__()
        self.name = "unique"

    def prop(self,inputs,structure,context):
        features = inputs[0].features
        scores = inputs[0].probs
        return SingleObject(features,scores/torch.sum(scores))

class CCount(DiffVertex):
    def __init__(self):
        super().__init__()
        self.name = "count"
    
    def prop(self,inputs,structure,context):
        return Rint(torch.sum(inputs[0].probs))

cimps = [CScene(),CMeasureColor(),CUnique(),CFilterColor(),CCount(),CMeasureRelation(),CFilterRelation()]

# write the executor to execute the program in the context
context = {"Objects":ObjectSet(torch.randn([3,OBJECT_FEATURE_DIM]),0.999 * torch.ones([3]))}
NORD = VertexExecutor(cstructure,cimps)

program = toFuncNode("count(filter_color(red,scene()))")
outputs = NORD.execute(program,context)
print(outputs.pdf(True))

programr = toFuncNode("filter_relation(unique(scene()),scene(),left)")
outputsr = NORD.execute(programr,context)
print(outputsr.pdf(True))

programr = toFuncNode("measure_relation(unique(scene()),unique(scene()))")
outputsr = NORD.execute(programr,context)
print(outputsr.pdf(True))

program = toFuncNode("measure_color(unique(scene()))")
outputs = NORD.execute(program,context)
print(outputs.pdf(True))

optim = torch.optim.Adam(nn.ModuleList([clist,rlist]).parameters(),lr = 2e-2)
for epoch in range(100):
    optim.zero_grad()
    program = toFuncNode("measure_color(unique(scene()))")
    outputs = NORD.execute(program,context)
    programr = toFuncNode("measure_relation(unique(scene()),unique(scene()))")
    outputsr = NORD.execute(programr,context)
    loss = 0 - NORD.supervise_prob(outputs,"green") 
    loss = loss -  NORD.supervise_prob(outputsr,"left")
    loss.backward();optim.step()
    if epoch%10 == 0 : print("Working Loss: ",dnp(loss))

print(outputs.pdf(True))
print(outputsr.pdf(True))