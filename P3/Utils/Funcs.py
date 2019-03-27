import numpy as np

def gauss_kernel(x1, x2, b):
    
    N_1 = x1.shape[0]
    N_2 = x2.shape[0]
    outer_elements = []
    
    for i in range(N_1):
        inner_elements = [np.exp((-1/b)*(np.sqrt(np.sum((x1[i,:] - x2[j,:])**2)))**2) for j in range(N_2)]
        outer_elements.append(inner_elements)
    
    outer_elements = np.array(outer_elements)
    return(outer_elements)
    

def gauss_predict(x_train, x_test, sig, b, y_train, y_test):
    N_test = x_test.shape[0]
    N_train = x_train.shape[0]
    
    kernel_test = gauss_kernel(x1 = x_test, x2 = x_train, b = b)
    kernel_train = gauss_kernel(x1 = x_train, x2 = x_train, b = b)
    
    pred = kernel_test.dot(np.linalg.inv(sig*np.identity(N_train) + kernel_train)).dot(y_train)
    
    RMSE = np.sqrt(np.sum((pred - y_test)**2) / N_test)
    
    return((pred, RMSE))