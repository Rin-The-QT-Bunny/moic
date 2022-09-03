
class Stack(object):
    def __init__(self,top = 0):
        self.top_pointer = top
        self.values = []
    def __str__(self):
        return "Stack:\n (pointer = {},value = {})".format(self.top_pointer,
                                                            self.top_pointer)
    def __eq__(self,o): return isinstance(o,Stack) and self.top_pointer == o.top_pointer

    def __ne__(self,o): return not self == o

class Queue(object):
    def __init__(self,front = 0,rear = -1,values = []):
        self.front = front
        self.rear = -1
        self.values = []
    def __str__(self): return "Queue:\n (front = %d,rear = %d)" %(self.front,self.rear)

    def __eq__(self,o): return  isinstance(o,Queue) and self.front == o.front and self.rear == o.rear and self.values == o.values

    def __ne__(self,o): return not self == o

class ChainNode(object):
    def __init__(self,value = None,prev = None,next = None):
        self.prev = prev
        self.next = next
        self.value = value
    def __str__(self): return "Chain Node: (prev = %s,next = %s,value = %s)" % (self.prev.content,self.next.content,self.content)

    def show(self): 
        if self.prev is None and self.next is None:
            return self.value
        if self.prev is None and self.next is not None:
            return "|%s->" % self.value
        if self.prev is not None and self.next is None:
            return "%s|" % self.value
        return self.prev.show() + "%s->%s" % (self.value,self.next.show())
    
    def __eq__(self,o): return isinstance(o,ChainNode) and self.value == self.value

    def __ne__(self,o): return not o == self

    def attach_to(self,next): self.next = next;next.prev = self

import networkx as nx
import numpy as np

class Graph(object):
    def __init__(self,nodes,edges,weights = None):
        self.nodes = nodes
        self.edges = edges
        self.weights = weights

    def __str__(self):
        return "Graph: " + "\n".join(self.nodes)

    def show(self,flag = False):
        F = nx.Graph()
        for node in self.nodes: F.add_node(node)
        return 0

    def build_edges(self):
        edges = {}
        N = len(self.nodes)
        for i in range(N):
            for j in range(N):
                if self.edges[i][j] == 1:
                    edges[self.nodes[i]] = self.nodes[j]
        return edges

    def build_matrix(self):
        N = len(self.nodes)
        connection = np.zeros([N,N])
        for i in range(N):
            from_node = self.nodes[i]
            to_node = self.weights[from_node]
            j = self.nodes.index(to_node)
            connection[i][j] = 1
        return connection

    def __eq__(self,o):
        if self.weights is None:
            return isinstance(o,self) and self.nodes == o.nodes and self.edges == o.edges 
        return isinstance(o,self) and self.nodes == o.nodes and self.edges == o.edges and self.weights == o.weights
    
    def __ne__(self,o):
        return not o == self

class BinaryTree(object):
    def __init__(self):
        super()

    def __eq__(self,o):
        return o == self
    
    def __ne__(self,o):
        return not o == self

class Treap(object):
    def __init__(self):
        super()
    def __eq__(self,o):return o==self

    def __ne__(self,o):return not o==self

class FuncNode:
    def __init__(self,token):
        self.token = str(token)
        self.content = token
        self.children = []
        self.father = []
        self.prior_length = None
        self.function = True
        self.type = "Function"
        self.isRoot = False
    def has_children(self): return len(self.children) != 0
    
    def has_args(self): return len(self.children) != 0
    
    def __str__(self):
        return_str = ""
        return_str += self.token + ""
        if (self.function):
            return_str += "("
        for i in range(len(self.children)):
            arg = self.children[i]
            # perform the unlimited sequence processing
            return_str += arg.__str__()
            max_length = len(self.children)
            if (self.prior_length != None):
                max_length = len(self.children)
            if (i < max_length -1):
                return_str += ","

        if (self.function):
            return_str += ")"
        return return_str
    
    def add_child(self,version_space):
        self.children.append(version_space)
        
    def add_token(self,token):
        vs = FuncNode(token)
        self.children.append(vs)

    def clear(self):self.children = []
    
    def length(self):
        score = 0
        if (len(self.children) > 0):
            for child in self.children:
                score += child.length()
        return 1 + score


def find_bp(inputs):
    loc = -1
    count = 0
    for i in range(len(inputs)):
        e = inputs[i]
        if (e == "("):
            count += 1
        if (e == ")"):
            count -= 1
        if (count == 0 and e == ","):
            return i
    return loc

def break_braket(program):
    try:
        loc = program.index("(")
    except:
        loc = -1
    if (loc == -1):
        return program,-1
    token = program[:loc]
    paras = program[loc+1:-1]
    return token,paras        

def break_paras(paras):
    try:
        loc = find_bp(paras)
    except:
        loc = -1
    if (loc == -1):
        return paras,True
    token = paras[:loc]
    rest = paras[loc+1:]
    return token,rest

def to_parameters(paras):
    flag = True
    parameters = []
    while (flag):
        token,rest = break_paras(paras)
        if (rest == True):
            flag = False
        parameters.append(toFuncNode(token))
        paras = rest
    return parameters

def toFuncNode(program):
    token,paras = break_braket(program)

    curr_head = FuncNode(token)
    if (paras == -1):
        curr_head.function = False
        curr_head.children = []
    else:
        children_paras = to_parameters(paras)
        curr_head.children = children_paras
    return curr_head
