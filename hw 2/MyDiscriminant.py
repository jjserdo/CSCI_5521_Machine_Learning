import numpy as np
import numpy.linalg as la


class GaussianDiscriminant:
    def __init__(self, k=2, d=8, priors=None, shared_cov=False):
        self.mean = np.zeros((k, d))  # mean
        self.shared_cov = (
            shared_cov  # using class-independent covariance or not
        )
        if self.shared_cov:
            self.S = np.zeros((d, d))  # class-independent covariance
        else:
            self.S = np.zeros((k, d, d))  # class-dependent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [
                1.0 / k for i in range(k)
            ]  # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        ''' Start Code '''
        class1 = Xtrain[ytrain == 1]
        class2 = Xtrain[ytrain == 2]
        self.mean[0] = np.mean(class1, axis=0) 
        self.mean[1] = np.mean(class2, axis=0) 
        ''' End Code '''

        if self.shared_cov:
            # compute the class-independent covariance
            ''' Start Code '''
            self.S = np.cov(Xtrain, rowvar=0, ddof=0)
            ''' End Code '''
        else:
            # compute the class-dependent covariance
            ''' Start Code '''
            self.S[0] = np.cov(class1, rowvar=0, ddof=0)
            self.S[1] = np.cov(class2, rowvar=0, ddof=0)
            ''' End Code '''

    def predict(self, Xtest):
        # predict function to get predictions on test set
        # Initialize predictions and discriminants
        predicted_class = np.ones(Xtest.shape[0])  
        g = np.zeros((Xtest.shape[0], self.k)) 

        for i in np.arange(Xtest.shape[0]):  # for each test set example
            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                if self.shared_cov:
                    ''' Start Code '''
                    # Calculate the Discriminant for Shared Covariance
                    x = Xtest[i] - self.mean[c]
                    Si = la.inv(self.S)
                    
                    g[i][c] =  -1/2 * np.dot(x.T, np.dot(Si, x))
                    g[i][c] +=        np.log(self.p[c])
                    ''' End Code '''
                else:
                    ''' Start Code '''
                    # Calculate the Discriminant for Different Covariance
                    x = Xtest[i] - self.mean[c]
                    Si = la.inv(self.S[c])
                    
                    g[i][c] =  -1/2 * np.log(la.det(self.S[c])) 
                    g[i][c] += -1/2 * np.dot(x.T, np.dot(Si, x)) 
                    g[i][c] +=        np.log(self.p[c])
                    ''' End Code '''
                    
        # determine the predicted class based on the values of discriminant function
        # remember to return 1 or 2 for the predicted class
            ''' Start Code '''
            if g[i][0] >= g[i][1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2
            ''' End Code '''
            
        return predicted_class
