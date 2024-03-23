import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,4])
y = np.array([2,4,1,9])

h = np.diff(x)

a1 = np.array([0,0,0,h[0]**3,0,0,3*(h[0]**2),0,6*h[0],0,0,0])
a2 = np.array([0,0,0,0,h[1]**3,0,0,3*(h[1]**2),0,6*h[1],0,0])
a3 = np.array([0,0,0,0,0,h[2]**3,0,0,0,0,0,3*(h[2]**2)])
b1 = np.array([0,0,0,h[0]**2,0,0,2*h[0],0,2,0,0,0])
b2 = np.array([0,0,0,0,h[1]**2,0,0,2*h[1],-2,2,0,0])
b3 = np.array([0,0,0,0,0,h[2]**2,0,0,0,-2,0,2*h[2]])
c1 = np.array([0,0,0,h[0],0,0,x[1],0,0,0,1,0])
c2 = np.array([0,0,0,0,h[1],0,-1*x[1],x[2],0,0,0,0])
c3 = np.array([0,0,0,0,0,h[2],0,-1*x[2],0,0,0,x[3]])
d1 = np.array([1,0,0,1,0,0,0,0,0,0,0,0])
d2 = np.array([0,1,0,-1,1,0,0,0,0,0,0,0])
d3 = np.array([0,0,1,0,-1,1,0,0,0,0,0,0])

y0 = np.array([y[0],y[1],y[2],0,0,y[3],0,0,0,0,0,0])

A = np.array([a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3]).T

x0= np.linalg.solve(A,y0)

coefficients = [
    (x0[0], x0[3], x0[6], x0[9]), 
    (x0[1], x0[4], x0[7], x0[10]),  
    (x0[2], x0[5], x0[8], x0[11])  
]
print(coefficients)

plt.scatter(x, y)
def piecewise_func(x):
    condlist =  [(x>=1)&(x < 2), (x >= 2) & (x < 3), (x >= 3) & (x<=4)]
    return np.piecewise(x, condlist, [lambda x: np.polyval(coefficients[0], x-1), 
                         lambda x: np.polyval(coefficients[1], x-2), 
                         lambda x: np.polyval(coefficients[2], x-3)])

x_range = np.linspace(0, 5, 400)

y_piecewise = piecewise_func(x_range)
plt.plot(x_range, y_piecewise, label='spline')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot with Points and Piecewise Function')
plt.legend()
plt.show()