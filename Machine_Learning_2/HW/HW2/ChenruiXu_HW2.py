##2
import matplotlib.pyplot as plt
import numpy as np

#plot the line
x = np.arange(-15, 5, 0.1)
y = 2*x**3+24*x**2-54*x 
plt.title("HW1.2")
plt.plot(x, y)

#calculate the derivative
from sympy import *
x = Symbol("x")
y = 2*x**3+24*x**2-54*x 
z1=diff(y,x)
#set=0 and solve it
solveset(z1,x)

z2=diff(z1,x)
solveset(z2,x)

##3
x=-9
y= 2*x**3+24*x**2-54*x
y

x=1
y= 2*x**3+24*x**2-54*x
y

x=-4
y= 2*x**3+24*x**2-54*x
y

x = np.arange(-3, 3, 0.1)
y = 2*x**3+24*x**2-54*x 
plt.title("HW1.2.1")
plt.plot(x, y)

x=-3
y= 2*x**3+24*x**2-54*x
y
 
x = np.arange(-999999, 0, 0.1)
y = 2*x**3+24*x**2-54*x 
plt.title("HW1.2.1")
plt.plot(x, y)

##4

#3D plot
from mpl_toolkits.mplot3d import Axes3D
fig1 = plt.figure()
ax = Axes3D(fig1)
x, y = np.mgrid[-5:5:50j, -5:5:50j]
z=x**2+y**2
plt.title("f(x, y) = x^2 + y^2")
ax.plot_surface(x, y, z, rstride=1, cstride=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#gradient vector
x,y=symbols('x,y')
z=x**2+y**2
v1=diff(z,x)
v2=diff(z,y)
print(v1)
print(v2)

#calculate gradient vector when plug in
def v(x,y):
    return([2*x,2*y])
#plug in
print(v(1,2))
print(v(2,1))
print(v(0,0))
 
##5
#3D plot
from mpl_toolkits.mplot3d import Axes3D
fig1 = plt.figure()
ax = Axes3D(fig1)
x, y = np.mgrid[-5:5:50j, -5:5:50j]
z=2*x*y+x**2+y
plt.title("f(x, y) = 2*x*y+x**2+y")
ax.plot_surface(x, y, z, rstride=1, cstride=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#gradient vector
x,y=symbols('x,y')
z=2*x*y+x**2+y
v1=diff(z,x)
v2=diff(z,y)
print(v1)
print(v2)

#calculate gradient vector when plug in
def v(x,y):
    return([2*x+2*y,2*x+1])
#plug in
print(v(1,1))
print(v(0,-1))
print(v(0,0))

##7
import numpy as np
a = np.array([[2, 0], [0, 5]])
b = np.array([[5, 1], [4, 5]])
c = np.array([[3, 5], [3, 1]])

#show both evalue and evactor
np.linalg.eig(a)
np.linalg.eig(b)
np.linalg.eig(c)

##9
A = np.array([[1,2,3],[1,0,1],[1,2,1]])
B= np.array([[0,1,0],[1,0,-1],[1,-1,1]])
C=np.array([[1,1],[1,-1]])
D = np.array([[1,2,2,1],[1,0,0,1],[3,4,4,3]])

#if the number of the rank of the matrix is smaller than the dimention of the matrix, it is not linear independent
np.linalg.matrix_rank(A)-A.shape[0]
np.linalg.matrix_rank(B)-B.shape[0]
np.linalg.matrix_rank(C)-C.shape[0]
np.linalg.matrix_rank(D)-D.shape[0]
