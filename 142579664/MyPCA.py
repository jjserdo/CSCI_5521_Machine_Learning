import numpy as np

def PCA(X,num_dim=None):
    # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
    ''' Start Code '''
    # Standardize matrix
    X_std = X - np.mean(X, axis=0)
    # Compute covariance matrix
    X_cov = np.cov(X_std.T)
    # Compute eigendecomposition
    X_eig_val, X_eig_vec = np.linalg.eig(X_cov)
    ## Sort eigenvalues and eigenvectors in descending order
    eig_sum = np.sum(X_eig_val)
    ''' End Code '''
    
    # select the reduced dimensions that keep >90% of the variance
    if num_dim is None:
        ''' Start Code '''
        # Calculate the appropriate num_dim if not assigned
        num_dim = 0
        per_var = np.sum(X_eig_val[:num_dim]) / eig_sum
        while per_var <= 0.9:
            num_dim += 1
            per_var = np.sum(X_eig_val[:num_dim]) / eig_sum  
        ''' End Code '''

    # project the high-dimensional data to low-dimensional one
    ''' Start Code '''
    eig_top = X_eig_vec[:,:num_dim]
    X_pca = X_std.dot(eig_top)
    ''' End Code '''

    return X_pca, num_dim
