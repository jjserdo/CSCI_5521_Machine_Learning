from matplotlib import pyplot as plt
import numpy as np


def plot_boundary(clf, x, y, p):
    """
    Plot the decision boundary of the kernel perceptron, and the samples (using different
    colors for samples with different labels)
    """    
    ##### Making the Contour Plot
    x1i = np.min(x.T[0])
    x1f = np.max(x.T[0])
    x2i = np.min(x.T[1])
    x2f = np.max(x.T[1])
    
    num = 100
    X1 = np.linspace(x1i,x1f,num)
    X2 = np.linspace(x2i,x2f,num)
    x1G, x2G = np.meshgrid(X1,X2)
    
    X = np.array(np.meshgrid(X1, X2)).T.reshape(-1, 2)
    
    c = clf.predict(X).reshape(len(X1),len(X2)).T
    
    fig, ax = plt.subplots()
    ax.contourf(x1G,x2G,c,levels=[-2,0,2])
    
    xneg = x[y == -1]
    xpos = x[y ==  1]
    
    ax.scatter(xpos.T[0], xpos.T[1], color='blue',zorder=1)
    ax.scatter(xneg.T[0], xneg.T[1], color='red',zorder=1)
    
    plt.show()
