import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

def calc_bernoulli(X):
    
    theta = np.sum(X) / X.shape[0]
    
    return(theta)

def calc_pareto(X):
    
    theta = X.shape[0] / np.sum(np.log(X))
    
    return(theta)

def predict_nbayes(pi_pred, out_1, out_0):
    
    out = (np.log(pi_pred) + np.sum(np.log(out_1))) > (np.log(1 - pi_pred) + np.sum(np.log(out_0)))
    
    return(out)

def calc_bayes(theta_0, theta_1, X_test, pi_pred):
    
    k = X_test.shape[0]
    out_0 = [theta_0[j]**X_test[j] * (1-theta_0[j])**X_test[j] if j < 54 else theta_0[j]*X_test[j]**(-theta_0[j]+1) for j in range(k)]
    out_1 = [theta_1[j]**X_test[j] * (1-theta_1[j])**X_test[j] if j < 54 else theta_1[j]*X_test[j]**(-theta_1[j]+1) for j in range(k)]
   
    out = predict_nbayes(pi_pred, out_1 = out_1, out_0 = out_0)
    return(out)


def plot_nbayes(weight_0, weight_1, save = False, save_name = "none"):
    fig,ax = plt.subplots(2,1)
    ax1,ax2=ax.ravel()
    markerline, stemlines, baseline = ax1.stem(range(1,58), weight_0, '-.')
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(baseline, 'color','r', 'linewidth', 2)
    ax1.set_title('Theta for y=0')

    markerline, stemlines, baseline = ax2.stem(range(1,58), weight_1, '-.')
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(baseline, 'color','r', 'linewidth', 2)
    ax2.set_title('Theta for y=1')

    plt.tight_layout()
    plt.xlabel('Feature')
    plt.ylabel('Estimated Weights')
    if save == True:
        plt.savefig(save_name)
    return(plt.show())


def K_nn(X_train, Y_train, X_test, k):
    
    prox = np.argsort(np.sum(np.abs(X_test - X_train), axis = 1))[:k]
    prox_2 = Y_train[prox, :]
    if np.mean(prox_2) > 0.5 :
        out = 1
    elif np.mean(prox_2) < 0.5 :
        out = 0
    else :
        out = np.random.choice([0,1])
    
    return(out)


def logistic_fit(X_train, Y_train, Iter = 1000, method = "SDG"):
    
    k = X_train.shape[1]
    W = np.zeros(k).reshape(k,1)
    liklihood = []
    Accuracy_train = []
    
    if method == "SDG":
        
        for i in range(Iter):
            
            eta = 1/(1e5*np.sqrt(i+2))
            term1 = expit(np.multiply(Y_train, X_train.dot(W)))
            liklihood.append(np.sum(np.log(term1 + 1e-10)))
            term2 = X_train.T.dot(np.multiply(Y_train,1-term1))

            W += term2*eta
            pred_y = np.sign(X_train.dot(W))
            acc = (sum(pred_y == Y_train))/len(pred_y)
            Accuracy_train.append(acc)
            
    elif method == "Newton":
        
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
    
    return((W, Accuracy_train, liklihood))
       
def logistic_predict(X_test, Y_test, fit = False, W = None, X_train = None, Y_train = None, Iter = 1000, method = "SDG"):
    
    Out = []
    
    if fit:
        W = logistic_fit(X_train = X_train, Y_train = Y_train, Iter = Iter, method = method)[0]
        pred_y = np.sign(X_test.dot(W))
        acc = (np.sum(pred_y == Y_test))/len(pred_y)
        Out = acc[-1]
    else :
        pred_y = np.sign(X_test.dot(W))
        acc = (np.sum(pred_y == Y_test))/len(pred_y)
        Out = acc
        
    return(acc)