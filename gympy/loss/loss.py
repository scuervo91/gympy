import numpy as np


def cross_entropy(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]
    print(AL.shape)
    print(Y.shape)
    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    cost = (-1/m)*(np.matmul(Y,np.log(AL.T)) + np.matmul((1-Y),np.log((1-AL.T))))
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    print(cost.shape)
    assert(cost.shape == ())
    
    return cost


def log_cost(AL,Y):
    m = Y.shape[1]
    
    j = (-Y * np.log(AL)) - ((1-Y)*np.log(1-AL))
    cost = (1/m)*j.flatten().sum()
    
    return cost