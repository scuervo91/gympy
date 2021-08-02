import numpy as np

def linear(w,x,b):
    return np.matmul(x,w) + b

def sigmoid(x):
    return 1 /( 1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanh_derivative(x):
    return 1 - np.power(tanh(x),2)

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return 1*(x>0)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))