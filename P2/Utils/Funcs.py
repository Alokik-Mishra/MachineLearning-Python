import numpy as np
import matplotlib.pyplot as plt

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
    
    