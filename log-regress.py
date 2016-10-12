import math
import random
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

# https://sourceforge.net/projects/scipy/files/scipy/0.16.1/

from pprint import pprint as pp
from sklearn import preprocessing


# http://www.scholarpedia.org/article/Seizure_prediction

#filename = './data/training_data_only_seizures.txt'
#filename = './data/training_data_hybrid.txt'
#filename = './data/training_data_seizures_plus_some_non.txt'
#filename = './data/training_data.txt' # non-seizure days and seizure days

#filename = './data/training_data.txt'
filename = './data/3_20_16-10_11_16.txt'
alldata = pd.read_csv(filename, usecols=[2,3,5], header=0, names=["DaysSince", "DoW", "WasSeizure"], delim_whitespace=False)

#filename = './data/ex2data1.txt'
#alldata = pd.read_csv(filename, header=None, names=["Ex1", "Ex2", "WasAdmit"], delim_whitespace=False)

# Set the training data
train_data = alldata

# Shuffle if we are going to split our training data
seed = int(time.time())
#seed = 123 # Fixed seed
alldata = alldata.sample(frac=1, random_state=seed).reset_index(drop=True)
m = len(alldata)

#train_data = alldata.iloc[m/4:, :] # Last 75%
#test_data = alldata.iloc[:m/4, :] # First 25%

train_data = alldata.iloc[:m*6/10, :] # First 60%
cv_data = alldata.iloc[m*6/10:m*8/10, :] # 60-80%
test_data = alldata.iloc[m*8/10:, :] # Last 20%

#train_data = alldata
#test_data = alldata.copy()

# Add bias (intercept) term
test_data.insert(0, 'Ones', 1)
cv_data.insert(0, 'Ones', 1)
train_data.insert(0, 'Ones', 1)

print("Examples: %s training, %s test, %s cross-val, %s total" % (
    len(train_data), len(test_data),
    len(cv_data), len(alldata)
    ))

print(alldata.head())

X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1:]

X = np.matrix(X.values)
y = np.matrix(y.values)

add_polys = False

if add_polys:
    for p in range(2,10):
        X = np.column_stack((X, np.power(X[:,0], p))) # Add polynomial power terms of DaysSince

Xtest = test_data.iloc[:,:-1]
ytest = test_data.iloc[:,-1:]

Xtest = np.matrix(Xtest.values)
ytest = np.matrix(ytest.values)

if add_polys:
    for p in range(2,10):
        Xtest = np.column_stack((Xtest, np.power(Xtest[:,0], p))) # Add polynomial power terms of DaysSince

Xcv = cv_data.iloc[:,:-1]
ycv = cv_data.iloc[:,-1:]

Xcv = np.matrix(Xcv.values)
ycv = np.matrix(ycv.values)

if add_polys:
    for p in range(2,10):
        Xcv = np.column_stack((Xcv, np.power(Xcv[:,0], p))) # Add polynomial power terms of DaysSince


# Initialize theta (random) -1 .. 1
theta = -1 + 2 * np.random.rand(1, X.shape[1])
theta = np.zeros((1, X.shape[1]))

def precision(true_pos, false_pos):
    if true_pos + false_pos == 0:
        return 0.0
    return true_pos / float(true_pos + false_pos)


def recall(true_pos, false_neg):
    if true_pos + false_neg == 0:
        return 0.0
    return true_pos / float(true_pos + false_neg)


def f1score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_cost(theta, X, y, reg_lambda):
    """
    Logistic cost function
    """
    # y = 1: should pred 1, so log(pred) = 0 (no penalty)
    # y = 0: should pred 0, so log(1 - pred) = 0 (no penalty)
    # as prediction deviates, the log cost increases towards infinity
    
    theta = np.matrix(theta)
    
    hx = X * theta.T # hx = pred
    
    first = np.multiply(-y, np.log(sigmoid(hx)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(hx)))
    
    # Regularize
    temp = theta
    temp[:,0] = 0  # only use theta 1 ... n (do not regularize 0)
    
    reg_cost = 0.5 * reg_lambda * np.asscalar(temp * temp.T) # Why asscalar needed?
    
    return (np.sum(first - second) + reg_cost) / len(X)


def predict(theta, X):
    X = np.matrix(X)
    theta = np.matrix(theta)
    prob = sigmoid(X * theta.T)
    return prob


def gradient(theta, X, y, reg_lambda):

    theta = np.matrix(theta)
    
    grad = np.zeros(int(theta.ravel().shape[1]))

    error = sigmoid(X * theta.T) - y
    
    # Regularization term, does not include j=0
    temp = theta
    temp[:,0] = 0

    reg = reg_lambda / len(X) * temp.T
    
    return X.T * error / len(X) + reg


# Minimize cost function
def learn(theta, lbda, X, y):
    result = opt.fmin_tnc(func=log_cost, x0=theta, fprime=gradient, args=(X, y, lbda), disp=False)
    return result

# Train parameters on the training set, with lamdda = 1 to learn some parameters
best_theta = learn(theta, 0.001, X, y)

# Now pick best lambda using the CV set
best_lbda = 0
best_cost = 100000
for lbda in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10):
    cost = log_cost(best_theta[0], Xcv, ycv, lbda)
    if cost < best_cost:
        best_lbda = lbda
        best_cost = cost
    print(lbda, cost)
    #preds = predict(result[0], test_set)
    #print(preds)

print 'Best: ', best_lbda, best_theta[0]

print "\n"

# And finally evaluate on our test set

#result = learn(theta, 1)
#print(log_cost(np.matrix([[0,0,0]]), X, y, 1))
#print(log_cost(result[0], X, y, 1))
#print(predict( result[0], np.matrix([[1],[45],[85]]).T));

threshold = 0.5

preds = [1 if x >= threshold else 0 for x in predict(best_theta[0], Xtest)]

for i, pred in enumerate(predict(best_theta[0], Xtest)):
    if y[i] == 1:
        print '{:.3f}'.format(float(pred)), y[i], Xtest[i][:,1], Xtest[i][:,2]
    

#print(preds)
#print(Xtest)
#print(ytest)

true_pos = [1 if (a == 1 and b == 1) else 0 for (a, b) in zip(preds, ytest)]
true_neg = [1 if (a == 0 and b == 0) else 0 for (a, b) in zip(preds, ytest)]
false_pos = [1 if (a == 1 and b == 0) else 0 for (a, b) in zip(preds, ytest)]
false_neg = [1 if (a == 0 and b == 1) else 0 for (a, b) in zip(preds, ytest)]

print 'TP {}'.format(sum(true_pos))
print 'FP {}'.format(sum(false_pos))
print 'TN {}'.format(sum(true_neg))
print 'FN {}'.format(sum(false_neg))

prec = precision(sum(true_pos), sum(false_pos))
rec = recall(sum(true_pos), sum(false_neg))
f1 = f1score(prec, rec)

print 'precision: {:.4f}'.format(prec)
print 'recall: {:.4f}'.format(rec)
print 'f1score: {:.4f}'.format(f1)

correct = sum(true_pos) + sum(true_neg)
accuracy = 100 * correct / float(len(Xtest))

print 'accuracy = {:.2f}%'.format(accuracy)  


#test_set =  [[1, 50, 50]]

# Todo: auto selection of polynomial terms
# Todo: add features (day of week)

# Compute the x and y coordinates for points on a sigmoid curve
#x = np.arange(-10, 10, 0.1)
#y = sigmoid(x)

# Plot the points using matplotlib
#plt.plot(x, y)
#plt.show()  # You must call plt.show() to make graphics appear.

