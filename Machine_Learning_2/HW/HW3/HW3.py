#%%
import math
import numpy as np
import pandas as pd
import scipy.linalg as la

#Q2
#%%
A=np.array([[-1,1,1],[1,1,1],[0,-2,0]])
AI=np.linalg.inv(A)
B=np.dot(AI,((np.array([[1,2,2]])).transpose()))
B

Q5
#%%
A_02=np.array([[1,-1],[1,1]])
EIG=la.eig(A_02)
print("eigenvalue is:",EIG[0])
print("eigenvalue is:",EIG[1])

