#%%
import numpy as np
import pandas as pd
import random
import math

#%%
s1=2
s2=10
x = np.arange(-2, 2, 0.01)
alpha=np.array([[0.1,0.2,0.5,1]])

#%% target formula
T=[]
def target(x):
    t = (math.e**(-abs(x)))*(math.sin(x*math.pi))
    T.append(t)
    return T

#???????????????????????????????
for i in range(len(x)):
    target(x[i])


#%%
T=np.transpose(np.array([T]))
X=np.transpose(np.array([x]))
#data=data.frame(T,X)

#%% initial parameters
#???????????????????????????????
def initial_parameters(s):
    np.random.seed(999)
    parameters={}
    W1= np.random.uniform(low=-0.5, high=0.5, size=(s,1))
    W2= np.random.uniform(low=-0.5, high=0.5, size=(1,s))
    b1=np.random.uniform(low=-0.5, high=0.5, size=(s,1)) 
    b2=np.random.uniform(low=-0.5, high=0.5, size=(1,1)) 

    # assert(W1.shape == (s, 1))
    # assert(b1.shape == (s, 1))
    # assert(W2.shape == (1, s))
    # assert(b2.shape == (1, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

#%%
initial_parameters(2)

#%%
def initialize_parameters_02(layer_dims):
    parameters={}
    L=len(layer_dims)

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.uniform(low=-0.5, high=0.5, size=(layer_dims[l],layer_dims[l - 1]))
        parameters['b' + str(l)] = np.random.uniform(low=-0.5, high=0.5, size=(layer_dims[l], 1))
        ### END CODE HERE ### W1= 
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

#%%
parameters = initialize_parameters_02([1,s1,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

#%% forward
def linear_forward(A,W,b):
    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

# %%
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

#%%
def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z 
    return A, cache

#%%
def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b) 
        A, activation_cache = sigmoid(Z)                 
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

#%%
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network ##????????????????

    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)],   activation='relu')
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A,parameters["W" + str(L)],parameters["b" + str(L)],activation="sigmoid")
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


#%%
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = - (1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    return cost

#%%
def linear_backward(dZ, cache): 
    A_prev, W, b = cache    
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m ) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db