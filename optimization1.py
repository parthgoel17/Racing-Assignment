import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
x = np.array([1,2,3,4])
y = np.array([2,4,1,9])

def spline_coefficients(slope_params,x,y):
	
	h = np.array([x[1]-x[0],x[2]-x[1],x[3]-x[2]])
	b = np.array([y[0], y[1], y[1], y[2], y[2], y[3], 0, 0, 0, 0, slope_params[0], slope_params[1]])
	b = b[:, np.newaxis]
	A = np.array([[0,0,0,1,0,0,0,0,0,0,0,0],[h[0]**3,h[0]**2,h[0],1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,h[1]**3,h[1]**2,h[1],1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,h[2]**3,h[2]**2,h[2],1],[3*h[0]**2,2*h[0],1,0,0,0,-1,0,0,0,0,0],[6*h[0],2,0,0,0,-2,0,0,0,0,0,0],[0,0,0,0,3*h[1]**2,2*h[1],1,0,0,0,-1,0],[0,0,0,0,6*h[1],2,0,0,0,-2,0,0],[0,2,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,6*h[2],2,0,0]])
	x0 = np.dot(np.linalg.inv(A), b)

	coefficients = [
	    (x0[0], x0[1], x0[2], x0[3]), 
	    (x0[4], x0[5], x0[6], x0[7]),  
	    (x0[8], x0[9], x0[10], x0[11])  
	]
	return coefficients


# def poly(a,b,c,d,x):
# 	ans = a*x**3 + b*x**2 + c*x + d
# 	return ans

# def poly1(a,b,c,x):
# 	ans = 3*a*x**2 + 2*b*x + c
# 	return ans

# def poly2(a,b,x):
# 	ans = 6*a*x + 2*b
# 	return ans

# def curvature(a,b,c,x):
# 	num = abs(poly2(a,b,x))/(1+(poly1(a,b,c,x)**2))**1.5
# 	return num

# def integrand(a,b,c,x):
#     return curvature(a,b,c,x)**2

# def sum_of_squares_of_curvatures(a, b, c, x0, x[0]):
#     sum_of_squares, _ = quad(integrand, x0, x[0], args=(a, b, c))
#    return sum_of_squares


def objective_function(slope_params, x, y):
    coefficients = spline_coefficients(slope_params, x, y)
    total_sum_of_squares = 0
    for i, coeff in enumerate(coefficients):
        def integrand(x):
            a, b, c, d = coeff  
            dydx = 3*a*x**2 + 2*b*x + c  
            d2ydx = 6*a*x + 2*b  
            curvature = abs(d2ydx) / ((1 + dydx**2)**1.5)
            return curvature**2

        segment_integral, _ = quad(integrand, x[i], x[i+1])
        total_sum_of_squares += segment_integral

    return total_sum_of_squares

def compute_gradient(slope_params, epsilon=1e-3):
    grad = np.zeros_like(slope_params)
    for i in range(len(slope_params)):
        slope_params_plus = np.array(slope_params)
        slope_params_plus[i] += epsilon
        obj_plus = -objective_function(slope_params_plus, x, y) 
        slope_params_minus = np.array(slope_params)
        slope_params_minus[i] -= epsilon
        obj_minus = -objective_function(slope_params_minus, x, y)
        
        grad[i] = (obj_plus - obj_minus) /  (2*epsilon)
        
    return grad

def gradient_descent(slope_params_initial, learning_rate, num_iterations):
    slope_params = np.array(slope_params_initial)
    for iteration in range(num_iterations):
        grad = compute_gradient(slope_params)
        slope_params = slope_params-(learning_rate * grad)
    
    return slope_params

initial_slope_params = [0, 0]  
learning_rate = 0.1 
num_iterations = 1000


optimized_slope_params = gradient_descent(initial_slope_params, learning_rate, num_iterations)
print("Optimized slope parameters:", optimized_slope_params)

coefficients = spline_coefficients(optimized_slope_params,x,y)
print(objective_function(optimized_slope_params,x,y))

def cubic(x, coeffs, x_i):
    a, b, c, d = coeffs
    return a*(x - x_i)**3 + b*(x - x_i)**2 + c*(x - x_i) + d


x_vals1 = np.linspace(1, 2, 100)
y_vals1 = cubic(x_vals1, coefficients[0], 1)

x_vals2 = np.linspace(2, 3, 100)
y_vals2 = cubic(x_vals2, coefficients[1], 2)

x_vals3 = np.linspace(3, 4, 100)
y_vals3 = cubic(x_vals3, coefficients[2], 3)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x_vals1, y_vals1, label='Cubic Spline Segment 1')
plt.plot(x_vals2, y_vals2, label='Cubic Spline Segment 2')
plt.plot(x_vals3, y_vals3, label='Cubic Spline Segment 3')

plt.scatter([1, 2, 3, 4], [2, 4, 1, 9], color='red') 
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cubic Spline Interpolation')
plt.legend()
plt.grid(True)
plt.show()