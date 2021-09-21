
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
class Draw:
    def __init__(self):
        y=None

    def multi_matrix_one_to_mul(self,input,weight,bias):
        y=np.multiply(weight,input)+bias
        return y

    def multi_matrix_multi_to_one(self,input,weight,bias):
        y=np.dot(weight,np.transpose(input))+bias
        return y

    def poslin(self,y):
        if y>0:
            y=y
        else:
            y=0
        return y

    def purelin(self,y):
        return y 

    def hardlim(self,y):
        if y>0:
            y=1
        else:
            y=0
        return y

    def hardlims(self,y):
        if y>0:
            y=1
        else:
            y=-1
        return y    

    def satlin(self,y):
        if y<0:
            y=0 
        elif y>1:
            y=1
        else:
            y=y
        return y

    def satlins(self,y):
        if y<-1:
            y=-1 
        elif y>1:
            y=1
        else:
            y=y
        return y

#%% Question1
W1=np.array([[-1,1]])
W2=np.array([[1,1]])
b1=np.array([[0.5,1]])
b2=np.array([[-1]])

X=np.arange(-2,2,0.05)

#%%
if __name__== "__main__":
    plot=Draw()

    Y_01=[]
    for x in X:
        X_01=np.array([[x,x]])
        y_01=plot.multi_matrix_one_to_mul(W1,X_01,b1)
        Y_01.append(y_01)
    
    A_01=[]
    for y in Y_01:
        a_01=plot.poslin(y[0][0])
        A_01.append(a_01)

    A_02=[]
    for y in Y_01:
        a_02=plot.poslin(y[0][1])
        A_02.append(a_02)

    A_03=[]
    for i in range(len(A_01)):
        a_03=np.array([[A_01[i],A_02[i]]])
        A_03.append(a_03)

    Y_02=[]
    for a in A_03:
        y_02=plot.multi_matrix_multi_to_one(a,W2,b2)
        Y_02.append(y_02)


# %%
P=[]
for i in range(len(Y_02)):
    p_01=Y_02[i][0][0]
    P.append(p_01)

plt.plot(X,P)

# %% Question2
X=np.arange(-2,2,0.05)

#%% 1
W1=np.array([[1]])
b1=np.array([[1]])
if __name__== "__main__":
    plot=Draw()

    Y_01=[]
    for x in X:
        y_01=plot.multi_matrix_one_to_mul(x,W1,b1)
        Y_01.append(y_01)

    A_01=[]
    for y in Y_01:
        a_01=plot.hardlims(y[0][0])
        A_01.append(a_01)
plt.plot(X,A_01)

#%% 2
W2=np.array([[-1]])
b2=np.array([[1]])
if __name__== "__main__":
    plot=Draw()

    Y_02=[]
    for x in X:
        y_02=plot.multi_matrix_one_to_mul(x,W2,b2)
        Y_02.append(y_02)

    A_02=[]
    for y in Y_02:
        a_02=plot.hardlim(y[0][0])
        A_02.append(a_02)
plt.plot(X,A_02)

# %% 3
W3=np.array([[2]])
b3=np.array([[3]])
if __name__== "__main__":
    plot=Draw()

    Y_03=[]
    for x in X:
        y_03=plot.multi_matrix_one_to_mul(x,W3,b3)
        Y_03.append(y_03)

    A_03=[]
    for y in Y_03:
        a_03=plot.purelin(y[0][0])
        A_03.append(a_03)
plt.plot(X,A_03) 

# %% 4
W4=np.array([[2]])
b4=np.array([[3]])
if __name__== "__main__":
    plot=Draw()

    Y_04=[]
    for x in X:
        y_04=plot.multi_matrix_one_to_mul(x,W4,b4)
        Y_04.append(y_04)

    A_04=[]
    for y in Y_04:
        a_04=plot.satlins(y[0][0])
        A_04.append(a_04)
plt.plot(X,A_04) 
# %% 5
W5=np.array([[-2]])
b5=np.array([[-1]])
if __name__== "__main__":
    plot=Draw()

    Y_05=[]
    for x in X:
        y_05=plot.multi_matrix_one_to_mul(x,W5,b5)
        Y_05.append(y_05)

    A_05=[]
    for y in Y_05:
        a_05=plot.poslin(y[0][0])
        A_05.append(a_05)
plt.plot(X,A_05) 

# %% Question 3
X=np.arange(-3,3,0.05)

# %% 1
W111=np.array([[2]])
b111=np.array([[2]])
if __name__== "__main__":
    plot=Draw()

    Y3_01=[]
    for x in X:
        y3_01=plot.multi_matrix_one_to_mul(x,W111,b111)
        Y3_01.append(y3_01)
P=[]
for i in range(len(Y3_01)):
    p_01=Y3_01[i][0][0]
    P.append(p_01)
plt.plot(X,P)

# %% 2
A_01=[]
for y in Y3_01:
    a_01=plot.satlins(y[0][0])
    A_01.append(a_01)
plt.plot(X,A_01)

# %% 3
W21=np.array([[1]])
b21=np.array([[-1]])
if __name__== "__main__":
    plot=Draw()

    Y_02=[]
    for x in X:
        y_02=plot.multi_matrix_one_to_mul(x,W21,b21)
        Y_02.append(y_02)
P=[]
for i in range(len(Y_02)):
    p_02=Y_02[i][0][0]
    P.append(p_02)
plt.plot(X,P)

# %% 4
A_02=[]
for y in Y_02:
    a_02=plot.satlins(y[0][0])
    A_02.append(a_02)
plt.plot(X,A_02)

# %% 5
X_03=[]
for i in range(len(A_01)):
    x_03=np.array([[A_01[i],A_02[i]]])
    X_03.append(x_03)

W211=np.array([[1,-1]])
b211=np.array([[0]])

Y_03=[]
for a in X_03:
    y_03=plot.multi_matrix_multi_to_one(a,W211,b211)
    Y_03.append(y_03)

P=[]
for i in range(len(Y_03)):
    p_01=Y_03[i][0][0]
    P.append(p_01)
plt.plot(X,P)

# %% 6
A3=[]
for y3 in Y_03:
    a3=plot.purelin(y3[0][0])
    A3.append(a3)
plt.plot(X,A3)


#%% Question 5
W=np.array([[1,1],[-1,1]])
b=np.array([[-2],[0]])

#%%
W1=W[0,:]
W2=W[1,:]
b1=b[0]
b2=b[1]

#%%
xx1=np.arange(-10,10,0.01)
yy1=(-W1[0]*xx1-b1)/W1[1]
plt.plot(xx1, yy1)

xx2=np.arange(-10,10,0.01)
yy2=(-W2[0]*xx1-b2)/W2[1]
plt.plot(xx2, yy2)

#%%
Q5=np.dot(W,np.array([[1,-1]]).T)+b
print(Q5)

#%%
plt.plot(xx1, yy1)
plt.plot(xx2, yy2)
plt.plot(1,-1,'or')

# %%
# Question 6
x = np.array([[1,4],[1,5],[2,4],[2,5],[3,1],[3,2],[4,1],[4,2]])
y = np.array([0,0,0,0,1,1,1,1])
w = np.array([1,1])
b = 1

#%%
def training(x, y, w, b, n_iter=10):
    E = np.sum(y-np.where((np.dot(w, x.T)+b)>=0, 1, 0))
    for j in range(n_iter):
        print('interation:', j)
        if E != 0:
            for i in range(len(x)):
                n = np.dot(w, x[i,:].T)+b
                a = np.where(n>=0, 1, 0)
                t = y[i]
                e = t - a

                if e!=0:
                    print('w before update:', w)
                    print('b before update:', b)
                    w = w + e*x[i,:]
                    b = b + e
                    print('w after update:', w)
                    print('b after update:', b)
            E = np.sum(y-np.where((np.dot(w, x.T)+b)>=0, 1, 0))
        else:
            print('final parameter: \t')
            print('w:', w)
            print('b:', b)
            break
    return w, b
        
# %%
W, B = training(x, y, w, b)

# %%
xxx=np.arange(-3,5,0.01)
yyy=(-W[0]*xxx-B)/W[1]
plt.plot(x.T[0],x.T[1],"or")
plt.plot(xxx, yyy)

# %%
