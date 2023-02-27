
from graphviz import Digraph
import math

class Value:
    def __init__(self,data, _children = (),_op ='',label=''):
        self.data  = data
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.label = label
        
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,_children=(self,other),_op='+')
        
        def _backward():
            self.grad += (1.0 * out.grad)
            other.grad += (1.0 * out.grad)
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self,other):
        return self + (-other)
    
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out =  Value(self.data * other.data,_children=(self,other),_op='*')
        
        def _backward():
            self.grad += (other.data * out.grad)
            other.grad += (self.data * out.grad)
        out._backward = _backward
        return out
    
    def __rmul__(self,other):
        return self * other
    
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out =  Value(self.data ** other,_children=(self,),_op=f'**{other}')
        
        def _backward():
            self.grad += (out.grad * other * self.data**(other-1))
        out._backward = _backward
        return out
    
    def __truediv__(self,other):
        return self * other**-1
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x),(self,),_op='exp')

        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out
    
    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s,(self,),_op='sigmoid')
        
        def _backward():
            self.grad =+ (s * (1-s)) * out.grad()
        out._backward = _backward
        return out
    
    def relu(self):
        x = self.data
        r = 0 if x < 0 else x
        out = Value(r,(self,),_op='relu')
        
        def _backward():
            self.grad =+ (out.data > 0) * out.grad()
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data 
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), _op='tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def trace(self):
        nodes, edges = set(), set()
        
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child,v))
                    build(child)
        build(self)
        return nodes,edges
    
    def draw_dot(self):
        dot = Digraph(
            format = 'svg',
            graph_attr = {'randir':'LR'}
        )
        
        nodes, edges = self.trace()
        for n in nodes:
            uid = str(id(n))
            dot.node(
                name = uid,
                label = "{%s | data %.4f | grad %.4f}" % (n.label,n.data,n.grad), shape='record'
            )
            if n._op:
                dot.node(name = uid + n._op,label = n._op)
                dot.edge(uid + n._op,uid)
                
        for n1,n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
                
        return dot
    
    def sort_topological(self):
        visited = set()
        topo = []
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
            return topo
        
        return build_topo(self)
    
    def backward(self):
        self.grad=1
        for node in reversed(self.sort_topological()):
            node._backward()
            
    