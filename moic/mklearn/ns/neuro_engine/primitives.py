import torch
import torch.nn as nn
import networkx as nx
from abc import abstractmethod

class CCG:
    def __init__(self,token,input_types,output_type):
        self.token = token
        self.input_types = input_types
        self.output_type = output_type
    def __str__(self):
        output = "CCG Name: {} \n Input: {} \n Output: {}".format(self.token,self.input_types,self.output_type)
        return output

class Observable(nn.Module):
    def __init__(self,semantics_dim):
        super().__init__()
        self.semantics = nn.Parameter(torch.randn([1,semantics_dim]))
        self.values = []

    def semantics_reprs(self):
        s = self.semantics
        return s/ torch.norm(s)

class Operator(nn.Module):
    def __init__(self,token,in_types,out_type,semantics_dim = 256):
        super().__init__()
        self.token = token
        self.semantics = nn.Parameter(torch.randn([1,semantics_dim]))
        self.args = nn.Parameter(torch.randn(len(in_types),semantics_dim))
        self.out_type = out_type
        self.in_types = in_types
        self.s_dim = semantics_dim

    def semantics_reprs(self):
        s = self.semantics
        return s

    def __str__(self):
        return "operator: " + self.token

class Constant(nn.Module):
    def __init__(self,token,type,semantics_dim = 256):
        super().__init__()
        self.token = token
        self.semantics = nn.Parameter(torch.randn([1,semantics_dim]))
        self.out_type = type
        self.s_dim = semantics_dim


    def semantics_reprs(self):
        s = self.semantics
        return s

    def __str__(self):
        return "constant: " + self.token 

class DSL(nn.Module):
    def __init__(self,operators,constants,s = 256):
        super().__init__()
        self.operators = operators
        self.constants = constants
        self.name = "default-domain-name"
        self.op_dim = s

    def display(self,token):
        F = nx.Graph()
        def connect(token):
            F.add_node(token)
            for i in range(len(self.constants)):
                const = self.constants[i]
                if (const.out_type == token):
                    F.add_node(const.token)
                    F.add_edge(token,const.token)
                    connect(const.token)
        connect(token)
        return F
    
    def distribution(self):
        tags = []
        semantics = []
        for op in self.operators:
            tags.append(op.token)
            semantics.append(op.semantics_reprs())
        for con in self.constants:
            tags.append(con.token)
            semantics.append(con.semantics_reprs())
        return tags,semantics
    
    def get_operator_args(self,operator_name):
        working_op = None
        for op in self.operators:
            if (op.token == operator_name):
                working_op = op
        return working_op.in_types,working_op.args

    def add_operators(self,operators):
        for operator in operators:
            self.operators.append(operator)
        return 0
    
    def add_constants(self,constants):
        for constant in constants:
            self.constants.append(constant)
        return 0
    
    def operators_of_type(self,Type):
        output_set = []
        for op in self.operators:
            if (op.out_type == "All" or op.out_type == Type or Type == "All"):
                output_set.append(op)
        return output_set

    def constants_of_type(self,Type):
        output_set = []
        for op in self.constants:
            if (op.out_type == "All" or op.out_type == Type or Type == "All"):
                output_set.append(op)
        return output_set
    
    def tokens_of_type(self,Type,seq):
        output_set = []
        for t in seq:
            if (t.out_type == "All" or t.out_type == Type or Type == "All"):
                output_set.append(t)
        return output_set
    
    def __str__(self):
        print("\n{} Operators in the DSL:".format(len(self.operators)))
        for op in self.operators:
            print(op)
        print("\n{} Constants in the DSL:".format(len(self.constants)))
        for cons in self.constants:
            print(cons)
        return "\nDomain Specific Language: " + self.name


class Vertex(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def propagate(self,inputs,context,structure):
        return 0


print("Quasi-Symbolic Concept Primitives Loaded.")