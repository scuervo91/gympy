from typing import Callable
import autograd.numpy as np
from pydantic import BaseModel, Field
from autograd import elementwise_grad as egrad


def categorical_cross_entropy(AL,Y): 
    return np.mean(-np.sum(Y*np.log(AL),axis=0))

categorical_cross_entropy_grad = egrad(categorical_cross_entropy,0)

def logistic_loss(AL,Y):
    m = Y.shape[1]
    return ((np.matmul(Y,np.log(AL.T)) + np.matmul((1-Y),np.log((1-AL.T)))))*(-1/m)

logistic_loss_grad = egrad(logistic_loss,0)

class LogisticLoss(BaseModel):
    forward: Callable[...,np.ndarray] = Field(logistic_loss,const=True) 
    backward: Callable[...,np.ndarray] = Field(logistic_loss_grad,const=True) 
    
    
class CategoricalCrossEntropy(BaseModel):
    forward: Callable[...,np.ndarray] = Field(categorical_cross_entropy,const=True) 
    backward: Callable[...,np.ndarray] = Field(categorical_cross_entropy_grad,const=True) 



    