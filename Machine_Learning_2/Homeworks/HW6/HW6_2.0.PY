#%%
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
# %% initial parameters
def initial_parameters(s):
    np.random.seed(999)
    W1= np.random.uniform(low=-0.5, high=0.5, size=(s,1))
    W2= np.random.uniform(low=-0.5, high=0.5, size=(1,s))#Different from W1 as they are not one kind of parameter
    b1=np.random.uniform(low=-0.5, high=0.5, size=(s,1))
    b2=np.random.uniform(low=-0.5, high=0.5, size=(1,1)) 
    
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}#put all parameters into one dictionary so that convinient to read
    return parameters
# %%
S1=2
S2=10
X = np.arange(-2, 2, 0.01)
Y=np.exp(-np.abs(X))*np.sin(np.pi*X) #Target
alpha=np.array([[0.1,0.2,0.5,1]])
X=X.reshape(len(X),1) #reshape into (400,1) matrix
Y=Y.reshape(len(Y),1) #reshape into (400,1) matrix
plt.plot(X,Y)

#%%
def Y_pred(X,parameters1):
    W1=parameters1["W1"]
    b1=parameters1["b1"]
    W2=parameters1["W2"]
    b2=parameters1["b2"]
    # print('w1:', parameters1["W1"].shape)
    # print('b1:', parameters1["b1"].shape)
    # print('w2:', parameters1["W2"].shape)
    # print('b2:', parameters1["b2"].shape)
    # print('sigmoid(W1*(X.T)+b1)', sigmoid(W1*(X.T)+b1).shape)
    # print('--------------------')
    #y_pred=np.dot(W2,sigmoid(W1*(X.T)+b1))+b2
    #y_pred=np.dot(W2,1/(1+np.exp(-(W1*(X.T)+b1))))+b2
    return np.dot(W2,1/(1+np.exp(-(W1*(X.T)+b1))))+b2

#%% batch
def batch_iteration(X,Y,batch_size=20,iteration_number=5000,lr=0.1,S_unit=S2):
    np.random.seed(999)
    number_of_batch=int(len(X)/batch_size)
    # W1=parameters1["W1"]
    # b1=parameters1["b1"]
    # W2=parameters1["W2"]
    # b2=parameters1["b2"]
    parameters=initial_parameters(S_unit)
    SE=[]
    for j in range(iteration_number): #how many iterations
        for n in range(number_of_batch):
            #initial_parameters
            W1=parameters["W1"]
            b1=parameters["b1"]
            W2=parameters["W2"]
            b2=parameters["b2"]


            #slice the whole data into batches
            X_batch=np.transpose(X[n*batch_size:(n+1)*batch_size,:])
            Y_batch=np.transpose(Y[n*batch_size:(n+1)*batch_size,:])

            #forward
            n1=np.dot(W1,X_batch)+b1 #(2,b)
            a1=1/(1+np.exp(-n1)) #(2,b)
            n2=np.dot(W2,a1)+b2 #(1,b)
            a2=n2

            #error
            e=Y_batch-a2

            #backward
            s2=-2*1*e #(1,b)
            #s1= np.dot(np.array([[W2[0][0],0],[0,W2[0][1]]]),(np.subtract(1,a1)*(a1)))*s2#(2,b)
            s1=W2.T*(np.subtract(1,a1)*(a1))*s2
            #update
            # print('b1:', parameters1["b1"])
            #print('e:', e)

            #Update parameters
            parameters["W1"]=W1-lr*(np.mean(s1*X_batch,axis=1)).reshape(S_unit,1)
            parameters["b1"]=b1-lr*(np.mean(s1, axis=1)).reshape(S_unit,1)
            parameters["W2"]=W2-lr*(np.mean(a1*s2,axis=1)).reshape(1,S_unit)
            parameters["b2"]=b2-lr*np.mean(s2)

        SE.append(np.dot(e,np.transpose(e))[0][0])
        y_pred=Y_pred(X,parameters)
        plt.plot(X,y_pred.T)
    
    plt.figure() 
    plt.plot(SE)
    return parameters

#%%
#batch_iteration(X,Y)

# %% Stochastic gradient approach(batch_size=1)
batch_iteration(X,Y,batch_size=1,iteration_number=5000,lr=0.2,S_unit=S1)
batch_iteration(X,Y,batch_size=1,iteration_number=5000,lr=0.2,S_unit=S2)

#%% Batch approach(batch_size=10/20/50)
batch_iteration(X,Y,batch_size=10,iteration_number=5000,lr=0.05,S_unit=S1)
batch_iteration(X,Y,batch_size=10,iteration_number=5000,lr=0.05,S_unit=S2)

# %%
batch_iteration(X,Y,batch_size=40,iteration_number=2000,lr=0.1,S_unit=20)


# %%
