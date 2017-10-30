"""
@author: Nicolas Loffreda
@date: 6/24/2017

Code to apply Principal Component Analysis (PCA) to a feature matrix
"""
import numpy as np
import matplotlib.pyplot as plt

def applyPCA(X, components=2, return_code='P'):
    '''
    Calculate the PCA of a feature matrix.
    Can be controled to set the number of PCA components we want to use.
    Will return P Matrix with dimension N x components
    '''
    if type(X) != np.ndarray:
        raise TypeError('Type of X must be a numpy array')
    if components > X.shape[1]:
        raise ValueError("Number of components can't be more than the number of features: " + str(X.shape[1]))
        
    # Normalize Feature Vector: Z
    mean_vec = X.mean(axis=0)
    Z = X - mean_vec

    # Covariance Matrix: C
    C = np.cov(X, rowvar=False)
    
    # EigenVectors: V
    eigvals, V = np.linalg.eigh(C)
    eigvals=np.flipud(eigvals)
    V=np.flipud(V.T)
    V = V[:components]
    
    # Principal Components matrix: P
    P = np.dot(Z,V.T)
    
    # Recuperated Matrix: X_Rec
    X_rec = np.dot(P, V) + mean_vec
    
    if return_code == 'P':
        return P
    if return_code == 'PX':
        return P, X_rec
    elif return_code == 'all':
        return mean_vec, Z, C, P, V, X_rec
    else:
        raise ValueError('Invalid return code: P, PX, all')
        
def eigvalsGraph(X, plot_comp=0):
    '''
    This function will calculate the eigvals variance contribution
    for a feature matrix and plot it
    '''
    if type(X) != np.ndarray:
        raise TypeError('Type of X must be a numpy array')
        
    # ToDo: Check that the sum of negative eigvals is not significant
    
    C = np.cov(X, rowvar=False)
    eigvals, V = np.linalg.eigh(C)
    eigvals_sum = sum(eigvals)
    eigvals_flip=np.flipud(eigvals)
    eigvals_partial_sum = 0
    eigvals_var_explain = []
    
    for i in eigvals_flip:
        eigvals_partial_sum += i
        eigvals_var_explain.append(eigvals_partial_sum)
        
    eigvals_var_explain = np.array(eigvals_var_explain)
    eigvals_var_explain = eigvals_var_explain / eigvals_sum
    ind = [i+1 for i in range(np.alen(eigvals_flip))]
    
    if plot_comp == 0:
        plt.plot(ind, eigvals_var_explain)
        plt.show()
    else:
        plt.plot(ind[:plot_comp], eigvals_var_explain[:plot_comp])
        plt.show()
        
    
        
            
    