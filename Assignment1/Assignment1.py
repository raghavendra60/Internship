import numpy as np
import matplotlib.pyplot as plt
from sympy import *
 
y1 = symbols('y₁')
 
def line_gen(A,B):
 len =10
 y_AB = np.zeros((2,len))
 lam_1 = np.linspace(0,1,len)
 for i in range(len):
   temp1 = A + lam_1[i]*(B-A)
   y_AB[:,i]= temp1.T
 return y_AB
 
A =np.array([2,-3])
B =np.array([10,y1])
C = 10
# Equation is ||A-B||=C
D = A-B
 
# S= list(solveset(Eq((A-B).dot(A-B),C**2),y1))
 
S= list(solveset(Eq(np.dot(np.transpose(D),D),C**2),y1))
 
print("The y₁ values are: ",S)
 
#converting to float
for j in range(len(S)):
 S[j]=round(float(S[j]),2)
 
print("The y₁ values in float are: ",S)
 
labels = ['Solution1','Soultion2']
j=0
for i in S:
 P1,P2 = np.array([2,-3]),np.array([10,i])
 y_FP = line_gen(P1,P2)
 plt.plot(y_FP[0,:],y_FP[1,:],label= labels[j])
 j = j + 1
 plt.plot(P1[0], P1[1], 'o')
 # plt.text(P1[0] * (1 - 0.1), P1[1] * (1 + 0.1), 'A')
 plt.text(P1[0] * (1 - 0.1), P1[1] * (1 - 0.5), '({}, {})'.format(P1[0], P1[1]))
 plt.plot(P2[0], P2[1], 'o')
 # plt.text(P2[0] * (1 - 0.1), P2[1] * (1 + 0.1), 'B')
 plt.text(P2[0] * (1 - 0.2), P2[1] * (1 + 0.2), '({}, {})'.format(P2[0], P2[1]))
 
y = np.linspace(-10, 5, 100)
 
# calculate the y value for each element of the x vector
x = y**2 + 6*y - 27 
ax = fig.add_subplot(1, 1, 1)
plt.grid()
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='lower left')
# plot the function
plt.plot(x,y, 'r',label=S)

# show the plot
plt.show()
fig, ax = plt.subplots()
ax.plot(x, y) 
plt.grid()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='lower left')
plt.axis('equal')
plt.show()
