import numpy as np
from numpy import linalg as la

"""
Specify your sigma for RBF kernel in the order of questions (simulated data, digit-49, digit-79)
"""
''' # mine '''
#sigma_pool = [0.1, 0.1, 0.1]
sigma_pool = 4 * np.ones(3)

class KernelPerceptron:
    """
    Perceptron Algorithm with RBF Kernel
    """

    def __init__(self, train_x, train_y, sigma_idx):
        self.sigma = sigma_pool[sigma_idx]  # sigma value for RBF kernel
        self.train_x = (
            train_x  # kernel perceptron makes predictions based on training data
        )
        self.train_y = train_y
        self.alpha = np.zeros([len(train_x),]).astype(
            "float32"
        )  # parameters to be optimized

    def RBF_kernel(self, x):
        # Implement the RBF kernel
        ''' # mine but didn't use' '''
        return np.exp(-np.norm(self.train_x-x)/self.sigma**2)

    def fit(self, train_x, train_y):
        # set a maximum training iteration
        max_iter = 1000

        # training the model
        for iter in range(max_iter):
            error_count = 0  # use a counter to record number of misclassification

            # loop through all samples and update the parameter accordingly
            ''' # mine '''
            for i in range(len(train_x)):
                y = 0
                for j in range(len(train_x)):
                    y += self.alpha[j] * self.train_y[j] * np.exp(-la.norm(self.train_x[j]-train_x[i])/self.sigma**2)
                y = np.sign(y)
                if np.sign(y) != train_y[i]:
                    self.alpha[i] += 1
                    error_count += 1
            # stop training if parameters do not change any more
            if error_count == 0:
                break

    def predict(self, test_x):
        # generate predictions for given data
        pred = np.zeros([len(test_x)]).astype("float32")  # placeholder
        ''' # mine '''
        for i in range(len(test_x)):
            y = 0
            for j in range(len(self.train_y)):
                y += self.alpha[j] * self.train_y[j] * np.exp(-la.norm(self.train_x[j]-test_x[i])/self.sigma**2)
            pred[i] = np.sign(y)
            
        return pred

    def param(
        self,
    ):
        return self.alpha
