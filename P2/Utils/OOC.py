import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.spatial import distance_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class Assignment2:    
    """
    Object for the second assignment. 
    Focusing on classification.
    Is inherited by the classes NaiveBayes, KNN, and LogisReg
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
    """
    Naive bayes model
    use_custom implments a coded from scratch version using bernoulli and pareto
    distribution appropriate weights. When use_custom is off we use the scikit
    learn GaussianNB
    """    
    def __init__(self, name, data, bernoulli_cols, pareto_cols, use_custom = True):        
        self.name = name
        self.data = data
        self.bernoulli_cols = bernoulli_cols
        self.pareto_cols = pareto_cols
        self.use_custom = use_custom
    
    def calc_bernoulli(self, X):
        """
        Function to calculate the ML estimate for bernoulli dist
        """
        theta = np.sum(X) / X.shape[0]
        return(theta)
    
    def calc_pareto(self, X):
        """
        Function to calculate the ML estimate for pareto dist
        """
        theta = X.shape[0] / np.sum(np.log(X))
        return(theta)

    def conditional_split(self):
        """
        Splits the training set based on the label
        """
        X_train = self.x_train
        Y_train = self.y_train
        self.x_train0 = X_train[(Y_train==0).reshape(Y_train.shape[0]), :]
        self.x_train1 = X_train[(Y_train==1).reshape(Y_train.shape[0]), :]
        
    def fit(self):
        """
        Fits the naive bayes (and calculates weights only in custom)
        For scikit we use the Gaussian naive bayes rather than the 
        bernoulli + pareto
        """
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
        """
        Get accuracy and y_hat based on the model fit
        """
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
            y_pred = list(map(int, y_hat))
            y_actual = Y_test.reshape(Y_test.shape[0]).tolist()
            self.accuracy = np.sum(np.array(y_pred) == np.array(y_actual))/len(y_pred)
        else:
            self.y_hat = self.model.predict(X_test_orig)
            self.accuracy = accuracy_score(self.y_test, self.y_hat)
        
    def conf_matrix(self):
        """
        Produces confusion matrix to compare TP, FP, TN, and TF
        """
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
        """
        Plots the weights for the custom mode only
        """
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
        

class KNN(Assignment2):    
    """
    K-nearest neighbors classifier:
    use_custom implements a coded from scratch version. When use_custom is 
    off, the scikit learn KNeighborsClassifier is used.
    """       
    def __init__(self, name, data, use_custom = True, dist_metric = 2):        
        self.name = name
        self.data = data
        self.use_custom = use_custom
        self.dist_metric = dist_metric
        
    def fit(self, k):
        """
        fit the model using the k nearest neighbors
        """
        X_train = self.x_train
        X_test = self.x_test
        Y_train = self.y_train
        Y_test = self.y_test
        p = self.dist_metric
        N = self.x_train.shape[0]
        self.k = k
        if self.use_custom:
            self.dist = distance_matrix(X_train, X_test, p = p).T
        else:
            model = KNeighborsClassifier(n_neighbors = k)
            model.fit(X = X_train, y = Y_train.reshape(N))
            self.model = model
        
    def predict(self):
        """
        predict the model based on the k neighbors speficied in the fit
        """
        X_train = self.x_train
        X_test = self.x_test
        Y_train = self.y_train
        Y_test = self.y_test
        N = X_test.shape[0]
        k = self.k
        if self.use_custom:
            y_pred = []
            for i in range(N):
                prox = np.argsort(np.sum(np.abs(X_test[i,:] - X_train), axis = 1))[:k]
                prox_2 = Y_train[prox, :]
                if np.mean(prox_2) > 0.5 :
                    out = 1
                elif np.mean(prox_2) < 0.5 :
                    out = 0
                else :
                    out = np.random.choice([0,1])
                y_pred.append(out)
            self.y_hat = y_pred
            self.accuracy = np.sum(np.array(self.y_hat) == Y_test.reshape(N)) / N
        else:
            model = self.model
            y_hat = model.predict(X_test)
            self.y_hat = y_hat
            self.accuracy = accuracy_score(self.y_test, self.y_hat)
            
    def conf_matrix(self):
        """
        Produces confusion matrix to compare TP, FP, TN, and TF
        """
        y_pred = self.y_hat        
        if self.use_custom:
            y_pred = list(map(int, y_pred))            
        else:
            pass        
        Y_test = self.y_test
        y_actual = Y_test.reshape(Y_test.shape[0]).tolist()
        y_pred_df = pd.DataFrame({'Actual': y_actual, 'Predicted' : y_pred})
        print(pd.crosstab(y_pred_df['Actual'], y_pred_df['Predicted']))
        
        
class LogisReg(Assignment2):   
    """
    K-nearest neighbors classifier:
    use_custom implements a coded from scratch version and allows for both traditional
    stochastic gradient descent (with a time increasing monotonic factor) and
    Newton's method. When use_custom is 
    off, the scikit learn LogisticRegression is used, and this is onlyu implemented using
    the Newton method.
    """
    def __init__(self, name, data, use_custom = True, method = "SGD"):        
        self.name = name
        self.data = data
        self.use_custom = use_custom
        self.method = method
    
    def fit(self, Iter, c = 1e9):
        """
        Fits the model using the training set. The number of iterations much be specified.
        a weigh factor c is used when implementing the scikit learn method, and the smaller
        the c, the bigger role the l2 penalty plays in fitting the model
        """
        X_train = self.x_train
        X_test = self.x_test
        Y_train = self.y_train
        Y_test = self.y_test
        N = Y_train.shape[0]
        if self.use_custom:
            k = X_train.shape[1]
            W = np.zeros(k).reshape(k,1)
            liklihood = []
            Accuracy_train = []
            if self.method == "SGD":
                for i in range(Iter):
                    eta = 1/(1e5*np.sqrt(i+2))
                    term1 = expit(np.multiply(Y_train, X_train.dot(W)))
                    liklihood.append(np.sum(np.log(term1 + 1e-10)))
                    term2 = X_train.T.dot(np.multiply(Y_train,1-term1))
                    W += term2*eta
                    pred_y = np.sign(X_train.dot(W))
                    acc = (sum(pred_y == Y_train))/len(pred_y)
                    Accuracy_train.append(acc)
            elif self.method == "Newton":
                for i in range(Iter):
                    eta = 1/(np.sqrt(i+2))
                    term1 = expit(np.multiply(Y_train, X_train.dot(W)))
                    liklihood.append(np.sum(np.log(term1 + 1e-10)))
                    dy =  X_train.T.dot(np.multiply(Y_train,1-term1))
                    dy_2 = -np.multiply(np.multiply(term1,1-term1),X_train).T.dot(X_train)
                    W -= np.linalg.inv(dy_2).dot(dy) * eta
                    pred_y = np.sign(X_train.dot(W))
                    acc = (np.sum(pred_y == Y_train))/len(pred_y)
                    Accuracy_train.append(acc)
            self.W = W
            self.liklihood = liklihood
            self.training_acc = Accuracy_train
        else:
            if self.method == "Newton":
                model = LogisticRegression(penalty = 'l2', C = c, solver = 'newton-cg')
                model.fit(X = X_train, y = Y_train.reshape(N))
                self.model = model
            elif self.method == "SDG":
                print("This optimization method cannot be implemented in sklearn, change to 'Newton'.")
            
        
    def predict(self):
        """
        predicts y_hat and provides model accuracy based on the fit method
        """
        X_test = self.x_test
        Y_test = self.y_test
        if self.use_custom:
            W = self.W
            N = Y_test.shape[0]
            pred_y = np.sign(X_test.dot(W))
            self.y_hat = np.array(pred_y)
            self.accuracy = np.sum(self.y_test == self.y_hat) / self.y_hat.shape[0]
        else:
            model = self.model
            self.y_hat = model.predict(X_test)
            self.accuracy = accuracy_score(self.y_test, self.y_hat)
              
    def conf_matrix(self):
        """
        Produces confusion matrix to compare TP, FP, TN, and TF
        """
        y_pred = self.y_hat       
        if self.use_custom:
            y_pred = list(map(int, y_pred))
        else:
            pass
        Y_test = self.y_test
        y_actual = Y_test.reshape(Y_test.shape[0]).tolist()
        y_pred_df = pd.DataFrame({'Actual': y_actual, 'Predicted' : y_pred})
        print(pd.crosstab(y_pred_df['Actual'], y_pred_df['Predicted']))

