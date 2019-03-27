# MachineLearning-Python
Re-implementation of the original R Machine Learning repo in Python with streamlined code. Models are both coded from scratch AND implemented using scikit-learn. Each assignment also has an "Object-oriented code" version, which allows for an easy comparision between the custom model and the scikit-learn version.

## Overview
1. [Course description](#desc)
2. [Tech/framework](#tech)
3. [Assignment 1](#as1)
4. [Assignment 2](#as2)
4. [Assignment 3](#as3)
4. [Assignment 4](#as4)
4. [Assignment 5](#as5)


<a name="desc"></a>
## Course Description
ELEN 4903 was a 1 semester class on machine learning theory. The course was almost fully theoretical, focusing on probability, optimization, and analysis. We covered most of the algorithms in use for the span of machine learning models, in a method grounded in rigorous mathematics. The goal of the class was to understand what is happening under the hood when using these models. The assignments however did involve some questions related to coding. All of the coding in this repo is done in R, the models are coded from scratch and all pre-made ML packages (caret, etc) are avoided. The topics covered in the course inclde:

* Regression
* Maximum Liklihood and Maximum a posteriori
* Regularization and the bias-variance tradeoff
* Classficiation
  * Naive bayes
  * K-nn
  * Perceptron and Logistic
  * Laplace approxmiation and Bayesian logistic
* Feature expansions and kernals
  * Kernalized perceptron
  * Kernalized knn
  * Gaussian processes
* SVMs
* Trees, bagging, and random forests
* Boosting
* The Expectation Maximization (EM) algorithm
* Clustering and Gaussian mixture models (GMMs)
* Matrix factorization and recommender systems
* Topic modelling and the LDA algorithm
* Nonnegative matrix factorization
* Principal components analysis (PCA)
* Markov chains
* Hidden Markov models (HMMs) and the Kalman filter

<a name="tech"></a>
## Tech/framework
All the coding has been done in python 3 using jupyter notebooks. The supplementary functions have been coded in .py files. All the models have been coded from scratch with some assistance from the scipy libraries. Matplotlib.pyplot is used for visualizations.

Each assignment has a section at the bottom where the results from teh coded-from-scratch models are compared with results from the popular ML library scikit-learn.

<a name="as1"></a>
## Assignment 1
The coding section of assignment one focused on regression and covered ridge regression and polynomial ridge regression.

The jupyter notebook can be seen [here](/P1/Assignment1.ipynb) and the .py file with many of the functions can be seen [here](/P1/Utils/Funcs.py), the OOC file can be seen [here](/P1/Utils/Funcs.py).


<a name="as2"></a>
## Assignment 2
The coding section of assignment two focused on classification and covered naive bayes, k-nearest neighbors, and logistic regression.

The jupyter notebook can be seen [here](/P2/Assignment2.ipynb) and the .py file with many of the functions can be seen [here](/P2/Utils/Funcs.py).

<a name="as3"></a>
## Assignment 3
The coding section of assignment three focuses on feature expansion kernal methods and ensemble methods. The Gaussian process and Adaboost boosting algorithms are covered.

The jupyter notebook can be seen [here](/P3/Assignment3.ipynb) and the .py file with many of the functions can be seen [here](/P3/Utils/Funcs.py).

<a name="as4"></a>
## Assignment 4
The coding section of assignment four changes the attention to focus on unsupervised methods, I cover k-means clustering and matrix factorization (reccomender system).

<a name="as5"></a>
## Assignment 5
The coding section of assignment five focuses on temporal and semi-supervised methods. I cover Markov chains and non-negative matrix factorisation (NMF).
