import autograd.numpy as np
from autograd import elementwise_grad as egrad

def linear(w,x,b):
    return np.matmul(w,x) + b

def sigmoid(x):
    return 1 /( 1 + np.exp(-x))

sigmoid_derivative = egrad(sigmoid)
# def sigmoid_derivative(x):
#     return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

tanh_derivative = egrad(tanh)
# def tanh_derivative(x):
#     return 1 - np.power(tanh(x),2)

def relu(x):
    return np.maximum(0,x)

relu_derivative = egrad(relu)

# def relu_derivative(x):
#     return 1*(x>0)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

softmax_derivative = egrad(softmax)