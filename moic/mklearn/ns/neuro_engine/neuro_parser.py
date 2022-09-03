import torch
import torch.nn as nn
from aluneth.rinlearn.nn.functional_net import *
from aluneth.data_structure import FuncNode
import numpy as np

class NeuroDecoder(nn.Module):
    def __init__(self,inputs_dim,DSL):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.projection_operator = FCBlock(256,2,inputs_dim,DSL.op_dim)
        self.repeater = FCBlock(256,2,DSL.op_dim + inputs_dim,inputs_dim)
        self.DSL = DSL
        self.working_loss = 0
        self.counter = 0

    def operator_pdf(self,key,Type):
        valid_operators = self.DSL.operators_of_type(Type)
        valid_constants = self.DSL.constants_of_type(Type)
        possible_operators = valid_operators
        possible_operators.extend(valid_constants)

        operator_features = []
        operator_keys = []
        require_args = []
        for op in possible_operators:
            operator_features.append(op.semantics_reprs())
            operator_keys.append(op.token)
            if (op.__class__.__name__ == "Operator"):
                require_args.append(True)
            else:require_args.append(False)
        operator_features = torch.cat(operator_features,0)
        key_proj = self.projection_operator(key) 
        #key_proj = key_proj/torch.norm(key_proj)
        #print("key proj:",key_proj)
        #pdf = torch.softmax(torch.matmul(key_proj,operator_features.permute(1,0)),1)
        pdf = torch.sigmoid( torch.cosine_similarity(key_proj,operator_features) /0.125)
        pdf = pdf/torch.sum(pdf)
        return operator_keys,pdf,require_args

    def forward(self,semantics,target_tree = None,track = False):
        flag = True
        if (target_tree != None):
            flag = False
        self.working_loss = 0
        self.counter = 0
        root = FuncNode("Root")
        traj = []
        def parse(head,semantics_key,Type):
            if(track):
                traj.append(self.projection_operator(semantics))
            keys,pdf,args_info = self.operator_pdf(semantics_key,Type)
            
            if (flag):
                max_index = torch.argmax(pdf)
            else:
                max_index = keys.index(target_tree[self.counter])
                self.counter += 1
            current_operator_name = keys[max_index]

            current_node = FuncNode(current_operator_name) # create the operator node
            current_node.function = args_info[max_index]
            head.children.append(current_node) # attach the node created
            self.working_loss += torch.log(pdf[max_index])
            # check whether there is a branching by the DSL grammar
            if ( args_info[max_index]):
                # start the branching sequence
                args_types,args_features = self.DSL.get_operator_args(current_operator_name)
                for i in range(len(args_types)):
                    arg = args_types[i]
                    #print(i,arg)
                    new_key = self.repeater(torch.cat([args_features[i:i+1],semantics_key],-1))
                    parse(current_node,new_key,arg)
            return 0,self.working_loss
        parse(root,semantics,"All")
        if (not track):
            return root,self.working_loss
        return root,self.working_loss,traj


class NeuroAttnDecoder(nn.Module):
    def __init__(self,inputs_dim,DSL):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.projection_operator = FCBlock(256,4,inputs_dim,DSL.op_dim)
        self.repeater = FCBlock(256,4,DSL.op_dim + inputs_dim,inputs_dim)
        self.DSL = DSL
        self.working_loss = 0
        self.counter = 0

    def operator_pdf(self,key,vecs,Type):
        valid_operators = self.DSL.operators_of_type(Type)
        valid_constants = self.DSL.constants_of_type(Type)
        possible_operators = valid_operators
        possible_operators.extend(valid_constants)

        operator_features = []
        operator_keys = []
        require_args = []
        for op in possible_operators:
            operator_features.append(op.semantics_reprs())
            operator_keys.append(op.token)
            if (op.__class__.__name__ == "Operator"):
                require_args.append(True)
            else:require_args.append(False)
        operator_features = torch.cat(operator_features,0)
        key_proj = self.projection_operator(key) 

        vecs = torch.tensor(vecs)
        #print(key_proj.shape,vecs.shape)

        scores = torch.sigmoid(torch.cosine_similarity(key_proj , vecs) * 7).unsqueeze(-1)
        #print(scores.shape,vecs.shape)
        key = torch.sum(scores * vecs,0)
        #key_proj = key_proj/torch.norm(key_proj)
        #print("key proj:",key_proj)
        #pdf = torch.softmax(torch.matmul(key_proj,operator_features.permute(1,0)),1)
        pdf = torch.sigmoid( (torch.cosine_similarity(key,operator_features)-0.0) /0.125)
        pdf = pdf/torch.sum(pdf)
        return operator_keys,pdf,require_args

    def forward(self,semantics,vectors,target_tree = None,track = False):
        flag = True
        if (target_tree != None):
            flag = False
        self.working_loss = 0
        self.counter = 0
        root = FuncNode("Root")
        traj = []
        def parse(head,semantics_key,Type):
            if(track):
                traj.append(self.projection_operator(semantics))
            keys,pdf,args_info = self.operator_pdf(semantics_key,vectors,Type)
            
            if (flag):
                max_index = torch.argmax(pdf)
            else:
                max_index = keys.index(target_tree[self.counter])
                self.counter += 1
            current_operator_name = keys[max_index]

            current_node = FuncNode(current_operator_name) # create the operator node
            current_node.function = args_info[max_index]
            head.children.append(current_node) # attach the node created
            self.working_loss += torch.log(pdf[max_index])
            # check whether there is a branching by the DSL grammar
            if ( args_info[max_index]):
                # start the branching sequence
                args_types,args_features = self.DSL.get_operator_args(current_operator_name)
                for i in range(len(args_types)):
                    arg = args_types[i]
                    #print(i,arg)
                    new_key = self.repeater(torch.cat([args_features[i:i+1],semantics_key],-1))
                    parse(current_node,new_key,arg)
            return 0,self.working_loss
        parse(root,semantics,"All")
        if (not track):
            return root,self.working_loss
        return root,self.working_loss,traj


class Reaver(nn.Module):
    def __init__(self,config,DSL):
        super().__init__()
        self.checker = None
        self.config = config
        self.s_dim = config.semantics_dim
        self.c_dim = config.concept_feature_base
        self.sc_joint_map = FCBlock(512,2,self.s_dim + self.c_dim,self.c_dim)
        self.DSL = DSL
        self.counter = 0
        self.keys = []
        self.vectors= []
        self.loss = 0
        
    def pdf_and_index(self,semantics_vector,operator_embedding,argmax = True):
        target = self.sc_joint_map(semantics_vector)
        simi_score = torch.sigmoid( (torch.cosine_similarity(target,operator_embedding)-0.2)/0.1 )
        pdf =  simi_score/torch.sum(simi_score)
        if (argmax):
            max_ind = torch.argmax(pdf)
        else:
            probs = pdf.detach().numpy()
            max_ind = np.random.choice(list(range(len(probs))),p = probs/np.sum(probs))
        return pdf, max_ind
    
    def forward(self,x,new_keys,new_vectors):
        x = x.reshape([1,-1])
        root = FuncNode("root")
        self.counter = 0
        self.keys,self.vectors = self.DSL.get_full_operators(new_keys,new_vectors)
        self.loss = 0
        def parse(semantics,node,arg):
            if (self.counter > 100):
                node.add_child(FuncNode("NA"))
                return 1
            pdf,max_ind = self.pdf_and_index(torch.cat([semantics,arg],-1),self.vectors,True)
            operator_token = self.keys[max_ind]
            
            working_node = FuncNode(operator_token)
            if (operator_token in self.DSL.operators):
                working_node.function = True
            else:
                working_node.function = False
            if (operator_token == "scene"):
                working_node.function = True
            node.add_child(working_node)
            self.loss = self.loss - torch.log(pdf[max_ind])
            self.counter += 1
            if (operator_token in self.DSL.operators):
                args = self.DSL.get_operator_args(operator_token)
                for i in range(args.shape[0]):
                    parse(semantics,working_node,args[i:i+1])
            return 0
        scope = torch.ones_like(x)
        parse(x,root,scope)
        return root,self.loss
    
    def evaluate(self,x,new_keys,new_vectors,target_head):
        x = x.reshape([1,-1])
        root = FuncNode("root")
        self.counter = 0



        self.keys,self.vectors = self.DSL.get_full_operators(new_keys,new_vectors)
        self.loss = 0
        def parse(semantics,node,arg,target_head):
            if (self.counter > 100):
                node.add_child(FuncNode("NA"))
                return 1

            pdf,max_ind = self.pdf_and_index(torch.cat([semantics,arg],-1),self.vectors)
            max_ind = self.keys.index(target_head.token)
            operator_token = self.keys[max_ind]
            working_node = FuncNode(operator_token)
            if (operator_token in self.DSL.operators):
                working_node.function = True
            else:
                working_node.function = False

            node.add_child(working_node)
            self.loss = self.loss - torch.log(pdf[max_ind])
            self.counter += 1
            
            if (operator_token in self.DSL.operators):
                args_p = self.DSL.get_operator_args(operator_token)
                size = args_p.shape[0]
                for i in range(size):
                    args_in = self.DSL.get_operator_args(operator_token)
                    parse(semantics,working_node,args_in[i:i+1],target_head.children[i])
            return 0
        scope = torch.ones_like(x)
        parse(x,root,scope,target_head)
        return root,self.loss