import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm

def predicter(mu_prev,Sigma_prev,u,A,B,R):
    mu_bar = np.dot(A,mu_prev) + np.dot(B,u)
    Sigma_bar = np.dot(np.dot(A, Sigma_prev), A.T) + R
    return (mu_bar,Sigma_bar)

def measurer(mu_prev,Sigma_prev,z,C,Q):
    mu_bar = z
    Sigma_bar = Q
    return (mu_bar,Sigma_bar)

def kalman_filter(mu_prev, Sigma_prev, u, z, A, B, C, R, Q):
    mu_bar = np.dot(A, mu_prev) + np.dot(B, u)
    Sigma_bar = np.dot(np.dot(A, Sigma_prev), A.T) + R
    
    if z is not None:
        K = np.dot(np.dot(Sigma_bar, C.T), np.linalg.inv(np.dot(np.dot(C, Sigma_bar), C.T) + Q))
        mu = mu_bar + np.dot(K ,z - (np.dot(C,mu_bar)))
        Sigma = np.dot((np.eye(len(mu)) - np.dot(K,C)),Sigma_bar)
    else:
        mu = mu_bar
        Sigma = Sigma_bar
    
    return mu, Sigma

A = np.array([[1]]) 
B = np.array([[1]])  
C = np.array([[1]])  
Q = 0.001  
R = 0.01


mu_corrected = np.array([0])  
Sigma_corrected = np.array([[0]])  
mu_predicted = np.array([0])
Sigma_predicted = np.array([[0]])
mu_measured, Sigma_measured = np.array([0]),np.array([[0]])
landmarks = np.array([3, 6, 9])
num_steps = 24
mu_list = []
Sigma_list = []

for step in range(num_steps):
    u = random.uniform(0.5,1)  
    z = None
    for landmark in landmarks:
        if abs(mu_corrected - landmark) < 0.4:  # Robot is close to a landmark
            z = np.array([landmark])
            break
    mu_predicted, Sigma_predicted = predicter(mu_predicted, Sigma_predicted, u,A,B,R)
    
    if z is not None:
        mu_corrected, Sigma_corrected = kalman_filter(mu_corrected, Sigma_corrected, u, z, A, B, C, R, Q)
        mu_measured,Sigma_measured = measurer(mu_measured,Sigma_measured,z,C,Q)
        mu_corrected_scalar = np.squeeze(mu_corrected)
        Sigma_corrected_scalar = np.squeeze(Sigma_corrected)
        mu_predicted_scalar = np.squeeze(mu_predicted)
        Sigma_predicted_scalar = np.squeeze(Sigma_predicted)

        x = np.linspace(mu_corrected_scalar - 3*np.sqrt(Sigma_corrected_scalar), 
                        mu_corrected_scalar + 3*np.sqrt(Sigma_corrected_scalar), 100)
        
        y1 = norm.pdf(x, mu_corrected_scalar, np.sqrt(Sigma_corrected_scalar))
        y2 = norm.pdf(x,mu_predicted_scalar,np.sqrt(Sigma_predicted_scalar))
        y3 = norm.pdf(x,mu_measured,np.sqrt(Sigma_measured))


        plt.plot(x, y1)
        plt.plot(x,y2)
        plt.plot(x,y3)
        plt.title('Gaussian Distribution')
        plt.xlabel('X')
        plt.ylabel('Probability Density')
        plt.show()

        mu_list.append(mu_corrected)
        Sigma_list.append(Sigma_corrected)
    else:
        mu_corrected, Sigma_corrected = kalman_filter(mu_corrected, Sigma_corrected, u, None, A, B, C, R, Q)
