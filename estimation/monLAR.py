# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:32:11 2020

@author: AurelieBoisbunon
"""

# Function for estimating the parameter beta of a linear model
# y = X*beta + error via the Lasso.
#
# Input:
# - X = design matrix (R^n*R^p),
# - y = study variable (R^n).
#
# Minimization problem of the Lasso:
# min_{beta} [1/2*||y-X*beta||^2 + lambda*|beta|],
# lambda >= 0
#
# This function begins by setting beta_chap to 0, then finds
# what is the minimum value of lambda such that one coefficient 
# of beta becomes non zero, i.e. such that one variable of X
# enters the subset of selected variables. 
# This process is repeated until lambda = 0 (giving the ordinary
# least-squares solution).
#
# We also compute here the least-squares estimate reduced to 
# the selection performed by Lasso.
#
# Output:
# - beta_LAR = Lasso estimate for beta on the regularization path
# - beta_LS  = Least-Squares estimate for beta on Lasso regularization path
# - lambda   = hyperparameter 
#

import numpy as np


def monLAR(X, y, normalize=False):
    
    n, p = X.shape
    if normalize:
        X = X-np.mean(X, axis=0)
        normX = np.sqrt(np.sum(np.power(X, 2), axis=0))
        X /= normX
        y = y-np.mean(y)
    
    if len(y.shape)>1:
        y = y.flatten()
    # Initialization
    indxsup = {}
    indxsup[0]	= []  # Selected variables
    I0 = {}
    I0[0] = [True for i in range(p)]             # Non selected variables
    beta_LAR = np.zeros((p, p+1))  # Lasso estimate for beta
    #beta_LS = np.zeros((p, p+1))  # LS estimate for beta
    
    r = np.dot(X.T, y)      # Correlations with y
    # Search for the 1st var to be added
    i1 = np.argmax(np.abs(r))
    lambda_ = {}
    lambda_[0] = abs(r[i1]) 
    indxsup[1] = [i1] 
    I0[1] = [i!=i1 for i in range(p)] 
    nI = 1        # length of indxsup 
    Xord = np.concatenate((X[:,indxsup[1]], X[:,I0[1]]), axis=1)
    M = np.dot(Xord.T, Xord) # Correlation matrix of X
    S = np.array(1/M[0, 0])  # Inverse of M
    v = np.dot(Xord.T, y)/lambda_[0]     # Subderivative of ||beta||_1
    
    k = 0 # Current step
    
    while (len(indxsup[k+1]) < p) and (lambda_[k]>=0):
        
        # Search for the next variable to be added
        tmp = np.dot(S, np.sign(v[:nI]))
        
        # Update of lambda
        # Should we add a variable?
        l1  = lambda_[k]*(v[nI:] -1)/(1-np.dot(M[nI:, :nI], tmp)) + lambda_[k]
        # Should we remove a variable?
        l1m = -lambda_[k]*(v[nI:] + 1)/(1+np.dot(M[nI:, :nI], tmp)) + lambda_[k]
        # Conditions on lambda
        lmb1 = l1[(l1<lambda_[k]) * (l1>=0)]
        lmb1m = l1m[(l1m<lambda_[k])*(l1m>=0)]        
        # Default if no value of l1 match the conditions
        lambda_[k+1] = lambda_[k]
        i_next = -2
        # Check if optimlity conditions are violated on lambda
        if len(lmb1)>0:
            lmb1 = np.max(lmb1)
            lambda_[k+1] = lmb1
            i_next = np.where(l1==lmb1)[0][0]
        if len(lmb1m)>0:
            lmb1m = np.max(lmb1m)
            if i_next==-2 or lmb1<lmb1m:
                lambda_[k+1] = lmb1m
                i_next = np.where(l1m==lmb1m)[0][0]
        
        l0  = lambda_[k] + beta_LAR[indxsup[k+1], k] /tmp
        l0n = l0[(l0<lambda_[k])*(l0>=0)]
        if len(l0n)>0 and lambda_[k+1]<np.max(l0n):
            # Search for the next variable to be deleted        
            i_next = -1
            lambda_[k+1] = np.max(l0n)
            i0 = np.where(l0==lambda_[k+1])[0][0] # Index of the variable to delete
        
        if i_next == -2:
            break
        # Update of beta
        if beta_LAR.shape[1] < k+2:
            beta_LAR = np.hstack((beta_LAR, np.zeros((p, 1))))
        beta_LAR[indxsup[k+1], k+1] = beta_LAR[indxsup[k+1], k] + (lambda_[k]-lambda_[k+1])*np.dot(S, np.sign(v[:nI]))
        
        if (k>0) and i_next==-1: # Step too long: one variable needs to be deleted
            
                # Update of step k and subsets 
                k = k + 1
                indxsup[k+1] = [indxsup[k][i] for i in range(len(indxsup[k]))]
                del indxsup[k+1][i0]
                I0[k+1] = [not(i in indxsup[k+1]) for i in range(p)]
                Xord = np.concatenate((X[:, indxsup[k+1]], X[:, I0[k+1]]), axis=1)
                nI = nI - 1
    
                # Update of mu=X*beta_chap, residuals, subderivative and correlation matrix of X 
                if(len(indxsup[k])==0):
                    mu = 0
                else:
                    mu = np.dot(X, beta_LAR[:, k])
                r = y - mu
                v = np.dot(Xord.T, r)/lambda_[k]
                S = reverseWoodBury(M[:nI+1, :nI+1], S, i0)
                M = np.dot(Xord.T, Xord)
    
                beta_LAR = np.hstack((beta_LAR, np.zeros((p, 2))))
                beta_LS = np.hstack((beta_LS, np.zeros((p, 2))))
        else:
            # Update of step k, subsets and inverse of M
            k = k + 1
            
            fact = 1/(M[nI+i_next, nI+i_next]-np.dot(np.dot(M[nI+i_next, :nI], S), M[:nI, nI+i_next]))
            S_M = np.dot(S, M[:nI, nI+i_next])
            if len(S_M.shape) == 1:
                S_M = S_M.reshape(-1, 1)
            S = np.hstack((S+fact*np.dot(S_M, S_M.T), -fact*S_M))
            S = np.vstack((S, np.hstack((-fact*S_M.T, [[fact]])) ))
            indxsup[k+1] = indxsup[k] + [np.where(I0[k])[0][i_next]]
            I0[k+1] = [not(i in indxsup[k+1]) for i in range(p)]
            Xord = np.concatenate((X[:, indxsup[k+1]], X[:, I0[k+1]]), axis=1)
            nI	= nI + 1
    
            # Update of mu=X*beta_chap, residuals, subderivative and correlation matrix of X 
            if (len(indxsup[k])==0):
                mu = 0
            else:
                mu = np.dot(X, beta_LAR[:, k])

            r = y - mu
            v = np.dot(Xord.T, r)/lambda_[k]
            M = np.dot(Xord.T, Xord)

        #beta_LS[indxsup[k], k]	= np.linalg.lstsq(X[:, indxsup[k]], y, rcond=None)[0]

    
    #beta_LS[indxsup[k+1], k+1]  = np.linalg.lstsq(X[:, indxsup[k+1]], y, rcond=None)[0]
    #if beta_LS.shape[1] > k+2:
    #    beta_LS = beta_LS[:, :k+2]
    beta_LAR[indxsup[k+1], k+1] = beta_LAR[indxsup[k+1], k] + lambda_[k]*np.dot(S, np.sign(v[:nI]))
    #if beta_LAR.shape[1] > k+2:
    #    beta_LAR = beta_LAR[:, :k+2]
    lambda_[k+1] = 0.
    
    return beta_LAR, beta_LS, lambda_

def reverseWoodBury(A, invA, ind=None):
#
#   calculate the inverse of the Matrix A(1:n-1,1:n-1) knowing A and A^(-1)
#   using The Woodbury Matrix Identity
#
#   A and invA are matrices of size n*n
#   The resulting matrix res is of size (n-1)*(n-1)
#   By J. Delporte and A. Boisbunon (Matlab code)  01/2012
#   Transcoded in Python:  02/2020
    
    n, p = A.shape
    if ind is not None:
        indices = [i for i in range(n)]
        del indices[ind]
        indices = indices + [ind]
        A = A[indices, :]
        A = A[:, indices]
        invA = invA[indices, :]
        invA = invA[:, indices]
    
    U = np.vstack( (np.hstack((np.zeros((n-1, 1)), A[:n-1, n-1].reshape(-1,1))),
                              np.hstack(([1.], (A[n-1, n-1]-1.)/2.)) ) )
    C = np.array([[0, 1], [1, 0]])
    invA_U = np.dot(invA, U)
    UT_invA = np.dot(U.T, invA)
    invCmUAU = np.linalg.inv(C - np.dot(U.T, invA_U))
    res = invA + np.dot(np.dot(invA_U, invCmUAU), UT_invA)
    res = res[:n-1, :n-1]

    return res

if __name__ == '__main__':
    
    import time
    
    np.random.seed(seed=1532)#
    n_train = 100
    p = 60
    X = np.random.normal(size=(n_train, p))
    kmax = np.random.randint(0, p/2, 1)[0]
    bTrue = np.zeros((p, 1))
    s = (np.unique(np.round((p-1)*np.random.random_sample((kmax, 1))))).astype(int)
    bTrue[s] = 10*np.random.random_sample((len(s), 1))-5
    sigma = 0.01
    y = np.dot(X, bTrue) + sigma*np.random.normal(size=(n_train, 1))
    
    start = time.time()
    beta_LAR, beta_LS, lambda_ = monLAR(X, y)
    print("monLAR: {0:.3f} s ".format(time.time() - start))
    
    from sklearn.linear_model import Lars
    
    start = time.time()
    reg = Lars(fit_path=True, fit_intercept=False)
    reg.fit(X, y)
    print("Lars: {0:.3f} s ".format(time.time() - start))
    coef_lars = reg.coef_path_
    alpha_lars = reg.alphas_*n_train
    
    from sklearn.linear_model import lasso_path
    start = time.time()
    alphas, coef_path, _ = lasso_path(X, y, alphas=np.array(list(lambda_.values()))/n_train)
    print("lasso_path: {0:.3f} s ".format(time.time() - start))
    if len(coef_path.shape)>2 and coef_path.shape[0]==1:
        coef_path = np.squeeze(coef_path)
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(beta_LAR.T, 'b', label="monLAR")
    plt.plot(coef_lars.T, '--g', label="LassoLars")
    plt.plot(coef_path.T, ':m', label="lasso_path")
    plt.tight_layout()