from aluneth.rinlearn.ns.neuro_engine.concept_structure import *
from aluneth.rinlearn.ns.neuro_engine.primitives import *
from aluneth.rinlearn.ns.neuro_engine.executor import *
from aluneth.data_structure import *
from aluneth.utils import *
from matplotlib.colors import rgb2hex

# implementations of DSL operators
class scene_imp(Vertex):
    def __init__(self):
        super().__init__()
        self.name = "cscene"
    def propagate(self,inputs,context,structure):
        obs = context["objects"]
        return {"outputs":obs,
            "logprobs":torch.log(0.99 * torch.ones([obs.shape[0],1]))}

class filter_imp(Vertex):
    def __init__(self):
        super().__init__()
        self.name = "cfilter"
    
    def propagate(self,inputs,context,structure):
        obs = inputs[0]
        scores = obs["logprobs"]
        value = inputs[1]
        outlp = []
        for i in range(obs["outputs"].shape[0]):
            Prb = structure.FilterClassify(EntityDot(obs["outputs"][i:i+1]),value)
            Prb = torch.log(Prb)
            outlp.append(torch.min(scores[i].reshape([1,-1]),Prb.reshape([1,-1])))
        outlp = torch.cat(outlp,0)

        return {"outputs":obs["outputs"],
                "logprobs":outlp.reshape([-1])}

class query_imp(Vertex):
    def __init__(self):
        super().__init__()
        self.name = 'cquery'
    def propagate(self,inputs,context,structure):
        object,category = inputs[0],inputs[1]
        object = torch.softmax(torch.log(object/(1-object)),0)

class unique_imp(Vertex):
    def __init__(self):
        super().__init__()
        self.name = 'cunique'
    def propagate(self,inputs,context,structure):

        object_scores = torch.exp(inputs[0]['logprobs'])
        object_scores = torch.softmax(torch.log(object_scores/(1-object_scores)),0)

        return {"outputs":inputs[0]["outputs"],"logprobs":torch.log(object_scores)}

class count_imp(Vertex):
    def __init__(self):
        super().__init__()
        self.name = "ccount"
    def propagate(self,inputs,context,structure):
        objects = inputs[0]
        scores = objects["logprobs"]
        return {'outputs':torch.sum(torch.exp(scores)),'logprobs':torch.sum(scores)}

class equal_imp(Vertex):
    def __init__(self):
        super().__init__()
        self.name = "cequal"
    def propagate(self,inputs,context,structure):
        states1 = inputs[0]
        states2 = inputs[1]
        scores1 = torch.exp(states1['logprobs'])
        scores2 = torch.exp(states2['logprobs'])

        scores1 = scores1.reshape([-1])
        scores2 = scores2.reshape([-1])


        true_prob = torch.dot(scores1 , scores2)
        
        true_prob = true_prob.reshape([1,1])

        false_prob = 1 - true_prob
       
        lps = [torch.log(true_prob),torch.log(false_prob)]

        return {"outputs":["True","False"],"logprobs":torch.cat(lps,0).reshape([-1])}

class measure_imp(Vertex):
    def __init__(self):
        super().__init__()
        self.name = "cmeasure"
    def propagate(self,inputs,context,structure):
        object_in = inputs[0]
        object_features = object_in['outputs']
        object_scores = torch.exp(object_in['logprobs'])
        measure_class = inputs[1]
        values = structure.concept_diction[measure_class]
        keys = []
        out_probs = []

        for val in values:
            keys.append(val.token)

        for target in keys:
            prob_target = 0
            for i in range(object_scores.shape[0]):
                entity = EntityDot(object_features[i:i+1])
                #print(object_scores[i].detach().numpy(),structure.FilterClassify(entity,target))
                prob_target = prob_target + object_scores[i] *structure.FilterClassify(entity,target).reshape([1])
            out_probs.append(prob_target)
        out_probs = torch.cat(out_probs,0)
        #print(keys,out_probs)
        out_probs = torch.log(out_probs)


        return {"outputs":keys,"logprobs":out_probs}


o_dim = 100

# constant values
cblue = Constant('cblue','color',o_dim)
cred = Constant('cred','color',o_dim)
cgreen = Constant('cgreen','color',o_dim)
cball = Constant('cball','category',o_dim)
ccube = Constant('ccube','category',o_dim)

# neuro operators
cscene = Operator('cscene',[],["ObjectSet"])
cfilter = Operator('cfilter',["ObjectSet"],["ObjectSet"])
ccount = Operator('ccount',["ObjectSet"],["Integer"])
cunique = Operator('cunique',["Object"],["ObjectSet"])
cequal = Operator('cequal',["Boolean"],["All","All"])
cmeasure = Operator('cmeasure',["color"],["Object","All"])

imps = torch.nn.ModuleList([scene_imp(),
                            filter_imp(),
                            count_imp(),
                            equal_imp(),
                            unique_imp(),
                            measure_imp()])


constants = nn.ModuleList([cblue,
                            cred,
                            cgreen,
                            cball,
                            ccube])

ops = nn.ModuleList([cscene,
                    cfilter,
                    ccount,
                    cunique,
                    cequal,
                    cmeasure])

cDSL = DSL(ops,constants)

# grammar of the concept structure
grammar = ConeConceptStructure(constants)
# hybrid executor using the conceptstructure and implementations
cexe = QuasiExecutor(grammar,imps)

print(cDSL) # signatures for the decoder to weave the program
print(cexe) # concept structuress and implementations based on this structure

# initiate the context scenario for the hybrid execution.
context = {"objects":nn.Parameter(torch.randn([1,o_dim]))}
context2 = {"objects":nn.Parameter(torch.randn([1,o_dim]))}
context3 = {"objects":nn.Parameter(torch.randn([1,o_dim]))}

cp = toFuncNode("cmeasure(cunique(cfilter(cscene(),cred)),color)")
cp2 = toFuncNode("cmeasure(cunique(cfilter(cscene(),cblue)),color)")
cpm = toFuncNode("cmeasure(cunique(cscene()),color)")
cpc = toFuncNode("cmeasure(cunique(cscene()),category)")
#print(cexe.concept_structure.ObjClassify(EntityBox(torch.randn([1,2]),2),"cblue"))

# detection initiate : torch.autograd.set_detect_anomaly(True)

data = [
    {"context":context,"program":cpm,"ground_truth":"cblue"},
    {"context":context2,"program":cpm,"ground_truth":"cblue"},
    {"context":context3,"program":cpm,"ground_truth":"cred"},
    {"context":context,"program":cpc,"ground_truth":"cball"},
    {"context":context2,"program":cpc,"ground_truth":"cball"},
    {"context":context3,"program":cpc,"ground_truth":"ccube"}
]

GroundConcepts(cexe,data) # Ground concepts on the scenes

def union_context(context1,context2):
    obs = [context1["objects"],context2["objects"]]
    return {"objects":torch.cat(obs,0)}

print("\nSoft Count Test:")
cp3 = toFuncNode("ccount(cfilter(cscene(),cred))")
r3 = cexe.run(cp3,union_context(union_context(context3,context2),context))
print(dnp(r3["outputs"]),int(0.5 + dnp(r3["outputs"])))

cp3 = toFuncNode("ccount(cfilter(cscene(),cblue))")
r3 = cexe.run(cp3,union_context(union_context(context3,context2),context))
print(dnp(r3["outputs"]),int(0.5 + dnp(r3["outputs"])))

cp3 = toFuncNode("ccount(cfilter(cscene(),cball))")
r3 = cexe.run(cp3,union_context(union_context(context3,context2),context))
print(dnp(r3["outputs"]),int(0.5 + dnp(r3["outputs"])))

cp3 = toFuncNode("ccount(cfilter(cscene(),ccube))")
r3 = cexe.run(cp3,union_context(union_context(context3,context2),context))
print(dnp(r3["outputs"]),int(0.5 + dnp(r3["outputs"])))

cp3 = toFuncNode("ccount(cfilter(cscene(),cgreen))")
r3 = cexe.run(cp3,union_context(union_context(context3,context2),context))
print(dnp(r3["outputs"]),int(0.5 + dnp(r3["outputs"])))

print("\nMeasurement on three objects:")
cp3 = toFuncNode("cmeasure(cunique(cscene()),color)")
r3 = cexe.run(cp3,context)
print(r3["outputs"],torch.exp(r3["logprobs"]).detach().numpy())

cp3 = toFuncNode("cmeasure(cunique(cscene()),color)")
r3 = cexe.run(cp3,context2)
print(r3["outputs"],torch.exp(r3["logprobs"]).detach().numpy())

cp3 = toFuncNode("cmeasure(cunique(cscene()),color)")
r3 = cexe.run(cp3,context3)
print(r3["outputs"],torch.exp(r3["logprobs"]).detach().numpy())

cp3 = toFuncNode("cmeasure(cunique(cscene()),category)")
r3 = cexe.run(cp3,context2)
print(r3["outputs"],torch.exp(r3["logprobs"]).detach().numpy())

cp3 = toFuncNode("cmeasure(cunique(cscene()),category)")
r3 = cexe.run(cp3,context3)
print(r3["outputs"],torch.exp(r3["logprobs"]).detach().numpy())

print("\nBoolean Equal Test:")
cp3 = toFuncNode("cequal(cmeasure(cunique(cfilter(cscene(),cred)),color),cmeasure(cunique(cfilter(cscene(),cblue)),color))")
r3 = cexe.run(cp3,union_context(union_context(context3,context2),context))
print(r3["outputs"],torch.exp(r3["logprobs"]).detach().numpy())

cp3 = toFuncNode("cequal(cmeasure(cunique(cfilter(cscene(),cred)),color),cmeasure(cunique(cfilter(cscene(),cred)),color))")
r3 = cexe.run(cp3,union_context(union_context(context3,context2),context))
print(r3["outputs"],torch.exp(r3["logprobs"]).detach().numpy())

cp3 = toFuncNode("cequal(cmeasure(cunique(cfilter(cscene(),ccube)),color),cmeasure(cunique(cfilter(cscene(),cball)),color))")
r3 = cexe.run(cp3,union_context(context3,context2))
print(r3["outputs"],torch.exp(r3["logprobs"]).detach().numpy())

cp3 = toFuncNode("cequal(cmeasure(cunique(cfilter(cscene(),cball)),color),cmeasure(cunique(cfilter(cscene(),cball)),color))")
r3 = cexe.run(cp3,union_context(union_context(context3,context2),context))
print(r3["outputs"],torch.exp(r3["logprobs"]).detach().numpy())

print("\nFilter Test:")
cp3 = toFuncNode("cfilter(cscene(),cblue)")
r3 = cexe.run(cp3,union_context(context3,context2))
print(torch.exp(r3["logprobs"]).detach().numpy())