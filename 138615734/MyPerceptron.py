"""
Refer to the pseudo code in our assignment, you can write code to complete your algorithm.

The following three parameter need to be returned:
    1. the final weight w
    2. the number of iteration that it makes w converge
    3. error rate, it represents the fraction of training samples that are classified to another one.
"""
# Hints
# the numpy package is useful for calculation

# Header
import numpy as np

# Please implement your algorithm
def MyPerceptron(X, y, w0=[0.1,-1.0]):
    # we initialize the variable to record the number of iteration that it makes w converge
    iter_value = 0
    w = w0
    
    # Iterate until convergence
    conv = False
    while conv == False: 
        conv = False
        for i in range(len(X)):
            if y[i] * np.dot(w, X[i]) <= 0:
                w = w + y[i]*X[i]
                conv = True
        iter_value += 1
    
    # Compute error rate
    error = 0
    for i in range(len(X)):
        if np.sign(np.dot(w, X[i])) != np.sign(y[i]):
            error += 1
    error_rate = error/len(X)

    return (w, iter_value, error_rate)
