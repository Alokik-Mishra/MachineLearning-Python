import numpy as np


def ridge_reg(X, y, lam):
    ''' 
    Ridge regression function
    Input: 
        X: feature matrix 
        y: label vector 
        d: diagonal with constant of d
       
    Output:
        out: optimized weights    
    '''
    
    k = X.shape[1]
    out = np.linalg.inv(lam*np.eye(k) + X.transpose().dot(X)).dot(X.transpose()).dot(y)
    return(out)

def df_lambda(lam, s):
    
    out = np.sum(np.divide((s**2), (lam + s**2)))
    return(out)

def predict(X, y, lam, X_test, Y_test):
    
    W = ridge_reg(X,y,lam)
    y_hat = X_test.dot(W)
    RMSE = np.sqrt(np.sum((Y_test - y_hat)**2) / Y_test.shape[0])
    
    return((y_hat, RMSE))

def poly_mat(X, p):
    X_main = X.copy()
    if p > 1:
        for j in range(2,p+1):
            X_j = np.power(X_main[:,:6], j)
            X = np.hstack((X_j, X))

    return(X)