import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

class Assignment1:
    
    """
    Object for the first week assignment. 
    Focusing on regression.
    Is inherited by the class RidgeReg
    """
    
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.decomposed = False
        self.poly = 1
        
    def make_array(self):
        """
        Converts the data into ndarrays
        """
        arrays = []
        for i in self.data:
            arrays.append(pd.read_csv("Data/" + i + ".csv", header = None).values)
        self.x_test = arrays[0]
        self.x_train = arrays[1]
        self.y_test = arrays[2]
        self.y_train = arrays[3]
    
    def get_shape(self):
        """
        Prints the numebr of features and size split between
        training and testing.
        """
        N_1, k_1 = self.x_train.shape
        N_2, k_2 = self.x_test.shape
        print("Number of features: " + str(k_1))
        print("Training sample: " + str(N_1))
        print("Testing sample: " + str(N_2))
        
    def svd(self):
        """
        Implements and captures the singular value decomposititon of 
        the training data
        """
        X_train = self.x_train
        self.X_u, self.X_s, self.X_v  = np.linalg.svd(X_train)
        self.decomposed = True
        
    def poly_expand(self, p):
        """
        Performs polynomial expansion on the training
        and testing features. Cross-terms are not included.
        """        
        X = self.x_train.copy()
        X_main = X.copy()
        if p > 1:
            for j in range(2,p+1):
                X_j = np.power(X_main[:,:6], j)
                X = np.hstack((X_j, X))
        self.x_train = X
        
        X = self.x_test.copy()
        X_main = X.copy()
        if p > 1:
            for j in range(2,p+1):
                X_j = np.power(X_main[:,:6], j)
                X = np.hstack((X_j, X))
        self.x_test = X
        self.poly = p
        
class RidgeReg(Assignment1):
    """
    Inherits the Assignment 1 class.
    Used for fitting and predicting ridge regression on a training and testing set.
    
    use_custom (default = True): when this is on, the model will be 
        implmented from scratch. when this is off it will use scikit-learn.
    """    
    def __init__(self, name, data, use_custom = True):
        self.use_custom = use_custom
        self.name = name
        self.data = data
        
    
    def fit(self, lam, use_stored = True, x_train = None, y_train = None):
        """
        Ridge regression function
        Performs ridge regression to fit the data and produce
        weights.
        
        use_stored (default = True): When true this will use the data from
        the class, when false, new data can be entered.
        """
        if self.use_custom:
            if use_stored:
                X = self.x_train
                y = self.y_train
            else:
                X = x_train
                y = y_train

            k = X.shape[1]
            W = np.linalg.inv(lam*np.eye(k) + X.transpose().dot(X)).dot(X.transpose()).dot(y)

            self.lam = lam
            self.W = W
            return W
        else:
            if use_stored:
                model = Ridge(lam)
                model.fit(self.x_train, self.y_train)
            else:
                model = Ridge(lam)
                model.fit(x_train, y_train)
            W = model.coef_
            self.lam = lam
            self.W = W
            self.model = model
            return W
    
    def df_lambda(self, use_stored = True, lam = None):
        """
        Computes df lambda based on value of lambda.
        Only works if using stored estimates from fit, otherwise
        is empty.
        """
        if use_stored:
            lam = self.lam
            s = self.X_s
            df_lam = np.sum(np.divide((s**2), (lam + s**2)))
            return df_lam
        else:
            pass
    
    def predict(self, use_stored = True, lam = None, x_train = None, y_train = None):
        """
        Produces the predicted labels (y) based on the
        testing set. If use_stored is on, the coefficients are pulled
        from the fit method. If it is off, new weights are computed using the 
        datasets entered as function attributes.
        """
        if self.use_custom:
            if use_stored:
                lam = self.lam
                W = self.W
            else :
                lam = lam
                W = fit(lam = lam, use_stored = False, x_train = x_train, y_train = y_train)
            X_test = self.x_test
            Y_test = self.y_test
            y_hat = X_test.dot(W)
            RMSE = np.sqrt(np.sum((Y_test - y_hat)**2) / Y_test.shape[0])

            return (y_hat, RMSE)
        else:
            if use_stored:
                lam = self.lam
                W = self.W
                y_hat = self.model.predict(self.x_test)
            else:
                lam = lam
                model = Ridge(lam)
                model.fit(x_train, y_train)
                y_hat = model.predict(self.x_test)
            RMSE = np.sqrt(mean_squared_error(self.y_test, y_hat))
            return(y_hat, RMSE)
            