#%%
import numpy as np
import random
import matplotlib.pyplot as plt
# %%
P = np.arange(-2, 2, 0.001)
T=np.exp(-np.abs(P))*np.sin(np.pi*P)
P=P.reshape(len(P),1)
T=T.reshape(len(T),1)
#%%
def main_function(P,T,batch_size,n_iteration,alpha,Hidden):
    np.random.seed(123)
    Error_square=[]
    W1=np.random.uniform(low=-0.5, high=0.5, size=(Hidden,1))
    W2=np.random.uniform(low=-0.5, high=0.5, size=(1,Hidden))
    b1=np.random.uniform(low=-0.5, high=0.5, size=(Hidden,1))
    b2=np.random.uniform(low=-0.5, high=0.5, size=(1,1)) 
    dict= {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    batch_count=int(len(P)/batch_size)
    for j in range(n_iteration):
        for i in range(batch_count):
            P_batch=np.transpose(P[i*batch_size:(i+1)*batch_size,:])
            T_batch=np.transpose(T[i*batch_size:(i+1)*batch_size,:])
            n1=np.dot(dict["W1"],P_batch)+dict["b1"]
            a1=1/(1+np.exp(-n1))
            n2=np.dot(dict["W2"],a1)+dict["b2"] 
            a2=n2
            e=T_batch-a2
            s2=-2*e
            s1=W2.T*(np.subtract(1,a1)*(a1))*s2
            dict["W1"]=dict["W1"]-alpha*(np.mean(s1*P_batch,axis=1)).reshape(Hidden,1)
            dict["b1"]=dict["b1"]-alpha*(np.mean(s1, axis=1)).reshape(Hidden,1)
            dict["W2"]=dict["W2"]-alpha*(np.mean(a1*s2,axis=1)).reshape(1,Hidden)
            dict["b2"]=dict["b2"]-alpha*np.mean(s2)
        Error_square.append(np.dot(e,np.transpose(e))[0][0])
        predictions=np.dot(dict["W2"],1/(1+np.exp(-(dict["W1"]*(P.T)+dict["b1"]))))+dict["b2"]
        plt.plot(P,predictions.T)  
    plt.figure() 
    plt.plot(Error_square)
    return dict
# %%
main_function(P,T,batch_size=1,n_iteration=5000,alpha=0.1,Hidden=2)
main_function(P,T,batch_size=1,n_iteration=5000,alpha=0.1,Hidden=10)
main_function(P,T,batch_size=40,n_iteration=5000,alpha=0.1,Hidden=2)
main_function(P,T,batch_size=40,n_iteration=5000,alpha=0.1,Hidden=10)
main_function(P,T,batch_size=40,n_iteration=50000,alpha=0.01,Hidden=10)
# %%
