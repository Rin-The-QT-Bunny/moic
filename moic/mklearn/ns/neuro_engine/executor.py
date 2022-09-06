import torch
import torch.nn as nn
from aluneth.data_structure import *
from aluneth.utils import *
class Executor(nn.Module):
    def __init__(self,concepts):
        super().__init__()
        self.operators = []
        self.grammar = None # grammar is the composition of concepts (attributes) 
        self.structure = None # structure is the composition of objects
        self.scenario = None
        self.concepts = nn.ModuleList(concepts)
    
    def add_operator(self,vertex):
        self.operators.append(vertex)
        return 0
    
    def add_operators(self,vertex_set):
        self.operators.extend(vertex_set)
        return 0

    def run(self,program_head,scene,ground_truth = "default"):
        def execute(head,scene):

            # find the name of the current operator
            operator_name = head.token
            if (operator_name == "scene" or operator_name == "Scene"):
                self.scenario = scene
                return [torch.zeros([scene.shape[0],1]),self.scenario]
            if (operator_name in self.grammar.concept_names):
                return operator_name
            # return the current scene 
            for bind in self.operators:
                # find the operator in the DSL that consists
                if bind.name == operator_name:
                    current_operator = bind
            # current operator is located
            
            curr_inputs = []
            # find arguments for the current operator
            for i in range(len(head.children)):
                curr_inputs.append(execute(head.children[i],scene))
            
            return current_operator.propagate(curr_inputs,self.grammar,scene)

        output = execute(program_head,scene)
        return output

class QuasiExecutor(nn.Module):
    def __init__(self,concept_structure,implementations):
        super().__init__()
        self.concept_structure = concept_structure
        self.implementations = implementations
    
    def run(self,program,context,ground_truth = None):

        def execute(head,context):
            # find the operator name for the current program head
            try:
                operator_name = head.content
            except:
                operator_name = head.token
            if (operator_name in self.concept_structure.concept_names):
                return operator_name
            
            if (len(head.children) == 0): # check if the parameter set is empty
                return []
            current_operator = None
            for vertex in self.implementations:
                if (vertex.name == operator_name):
                    current_operator = vertex
            if (current_operator == None):
                print("current_operator implementation {} not found".format(operator_name))
            # locate the current operator implementation by find the the operator name == implementation.name
            curr_inputs = []
            for child in head.children:
                curr_inputs.append(execute(child,context))

            return current_operator.propagate(curr_inputs,context,self.concept_structure)
        outputs = execute(program,context)
        return outputs

print("Quasi-Symbolic Executor Loaded")


def GroundConcepts(executor,ground_data,lr = 2e-3):
    optim = torch.optim.Adam(executor.parameters(),lr)
    for epoch in range(2400):
        optim.zero_grad()
        Loss = 0
        for bind in ground_data:
            context,program,ground_truth = bind["context"],bind["program"],bind["ground_truth"]
            if program.__class__.__name__ != "FuncNode":
                program = toFuncNode(program)

            result = executor.run(program,context)
            outputs,logprobs = result["outputs"],result["logprobs"]
            loss_index = outputs.index(ground_truth)
            Loss = Loss - logprobs[loss_index]
        if ((epoch + 1) % 600 == 0):
            print("Epoch: {} Loss: {}".format(epoch + 1,dnp(Loss)))
        Loss.backward()
        optim.step()
    print("Ground Concepts Finished with residual of {}".format(dnp(Loss)))