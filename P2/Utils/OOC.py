import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

class Assignment2:
    
    """
    Object for the first week assignment. 
    Focusing on regression.
    Is inherited by the class RidgeReg
    """
    
    def __init__(self, name, data):
        self.name = name
        self.data = data

        
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
        
    def add_ones(self):
        """
        Adds a constant columns of 1s to training
        and testing features.
        """
        N_train = self.x_train.shape[0]
        N_test = self.x_test.shape[0]

        self.x_train = np.column_stack((self.x_train, np.ones((N_train,1))))
        self.x_test = np.column_stack((self.x_test, np.ones((N_test,1))))

    def zero_to_neg(self):
        """
        Converting 0s in output var to -1
        """
        self.y_train[self.y_train == 0] = -1
        self.y_test[self.y_test == 0] = -1
        
        
class NaiveBayes(Assignment2):
    
    def __init__(self, name, data, bernoulli_cols, pareto_cols, use_custom = True):        
        self.name = name
        self.data = data
        self.bernoulli_cols = bernoulli_cols
        self.pareto_cols = pareto_cols
        self.use_custom = use_custom
    
    def calc_bernoulli(self, X):    
        theta = np.sum(X) / X.shape[0]
        return(theta)
    
    def calc_pareto(self, X):
        theta = X.shape[0] / np.sum(np.log(X))
        return(theta)

    def conditional_split(self):
        X_train = self.x_train
        Y_train = self.y_train
        self.x_train0 = X_train[(Y_train==0).reshape(Y_train.shape[0]), :]
        self.x_train1 = X_train[(Y_train==1).reshape(Y_train.shape[0]), :]
        
    def fit(self):
        
        if self.use_custom:
            bern_cols = self.bernoulli_cols
            pareto_cols =  self.pareto_cols
            X_0 = self.x_train0
            X_1 = self.x_train1
            theta_0 = [self.calc_bernoulli(X_0[:,i]) for i in range(bern_cols[0],bern_cols[1])]
            theta_0.extend([self.calc_pareto(X_0[:,i]) for i in range(pareto_cols[0],pareto_cols[1]-1)])

            theta_1 = [self.calc_bernoulli(X_1[:,i]) for i in range(bern_cols[0],bern_cols[1])]
            theta_1.extend([self.calc_pareto(X_1[:,i]) for i in range(pareto_cols[0],pareto_cols[1]-1)])

            self.theta0 = theta_0
            self.theta1 = theta_1
            
        else:
            Y_train_scikit = self.y_train.reshape(self.y_train.shape[0])
            model = GaussianNB()
            model.fit(X = self.x_train, y = Y_train_scikit)
            self.model = model
        
    def predict(self):
        X_test_orig = self.x_test
        if self.use_custom:
            Y_test = self.y_test
            theta_0 = self.theta0
            theta_1 = self.theta1
            bern_cols = self.bernoulli_cols
            pareto_cols =  self.pareto_cols
            N,k = self.x_test.shape
            pi_pred = np.mean(Y_test)
            y_hat = []
            for i in range(N):
                X_test = X_test_orig[i,:]
                out_0 = [theta_0[j]**X_test[j] * (1-theta_0[j])**X_test[j] 
                         if j in range(bern_cols[0],bern_cols[1]) 
                         else theta_0[j]*X_test[j]**(-theta_0[j]+1) for j in range(k)]

                out_1 = [theta_1[j]**X_test[j] * (1-theta_1[j])**X_test[j] 
                         if j in range(bern_cols[0],bern_cols[1])
                         else theta_1[j]*X_test[j]**(-theta_1[j]+1) for j in range(k)]

                y_hat.append((np.log(pi_pred) + np.sum(np.log(out_1))) > (np.log(1 - pi_pred) + np.sum(np.log(out_0))))

            self.y_hat = y_hat

        else:
            self.y_hat = self.model.predict(X_test_orig)
        
    def conf_matrix(self):
        y_pred = self.y_hat
        
        if self.use_custom:
            y_pred = list(map(int, y_pred))
            
        else:
            pass
        
        Y_test = self.y_test
        y_actual = Y_test.reshape(Y_test.shape[0]).tolist()
        y_pred_df = pd.DataFrame({'Actual': y_actual, 'Predicted' : y_pred})
        
        print(pd.crosstab(y_pred_df['Actual'], y_pred_df['Predicted']))
        
    def plot_weights(self, save = False, filename = "None", dims = (12,4)):
        if self.use_custom:
            weight_0 = self.theta0
            weight_1 = self.theta1
        else:
            weight_0 = self.model.coef_[0]
            weight_1 = self.model.coef_[1]
        plt.figure(1, figsize=dims, dpi=80, facecolor='w', edgecolor='k')
        plt.subplot(1,2,1)
        markerline, stemlines, baseline = plt.stem(range(1,58), weight_0, '-.')
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.setp(baseline, 'color','r', 'linewidth', 2)
        plt.title('Theta for y=0')
        
        plt.subplot(1,2,2)
        markerline, stemlines, baseline = plt.stem(range(1,58), weight_1, '-.')
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.setp(baseline, 'color','r', 'linewidth', 2)
        plt.title('Theta for y=1')

        plt.tight_layout()
        plt.xlabel('Feature')
        plt.ylabel('Estimated Weights')
        if save == True:
            plt.savefig(filename)
        print(plt.show())
        
        