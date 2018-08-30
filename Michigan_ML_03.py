# coding: utf-8

# ---
#
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
#
# ---

# # Assignment 3 - Evaluation
#
# In this assignment you will train several models and evaluate how effectively they predict instances of fraud using data based on [this dataset from Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud).
#  
# Each row in `fraud_data.csv` corresponds to a credit card transaction. Features include confidential variables `V1` through `V28` as well as `Amount` which is the amount of the transaction. 
#  
# The target is stored in the `class` column, where a value of 1 corresponds to an instance of fraud and 0 corresponds to an instance of not fraud.

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt


# ### Question 1
# Import the data from `fraud_data.csv`. What percentage of the observations in the dataset are instances of fraud?
#
# *This function should return a float between 0 and 1.*

# In[ ]:

def answer_one():
    df = pd.read_csv('fraud_data.csv')
    np.mean(df['Class'])*100
    return np.bincount(df['Class'])[1]/np.bincount(df['Class']).sum()*100

answer_one()

# In[ ]:

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ### Question 2
#
# Using `X_train`, `X_test`, `y_train`, and `y_test` (as defined above), train a dummy classifier that
# classifies everything as the majority class of the training data. What is the accuracy of this classifier?
# What is the recall?
#
# *This function should a return a tuple with two floats, i.e. `(accuracy score, recall score)`.*

# In[ ]:

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import confusion_matrix

    # Fit DummyClassifier
    dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)

    # Predict
    y_dummy_predictions = dummy_majority.predict(X_test)
    # np.bincount(y_dummy_predictions)

    # Get scores (manually)
    cm = confusion_matrix(y_test, y_dummy_predictions)
    print(cm)
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    accuracy_sc = float((TN + TP) /(TN + FP + FN + TP))
    # precision_sc = float(TP /(TP + FP))
    recall_sc = float(TP /(TP + FN))        # TPR
    # specificity_sc = float(FP / (FP + TN))  # FPR

    # Sanity check
    assert type(accuracy_sc) == type(0.0), str(type(accuracy_sc))
    assert type(recall_sc) == type(0.0), str(type(recall_sc))

    return accuracy_sc, recall_sc

answer_two()

# ### Question 3
#
# Using X_train, X_test, y_train, y_test (as defined above),
# train a SVC classifer using the default parameters.
# What is the accuracy, recall, and precision of this classifier?
#
# *This function should a return a tuple with three floats, i.e. `(accuracy score, recall score, precision score)`.*

# In[ ]:

def answer_three():
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    from sklearn.svm import SVC

    svm = SVC().fit(X_train, y_train)
    y_predicted = svm.predict(X_test)

    accuracy_sc = accuracy_score(y_test, y_predicted)
    recall_sc = recall_score(y_test, y_predicted)
    precision_sc = precision_score(y_test, y_predicted)

    # Sanity check
    # print('Accuracy: {:.3f}'.format(accuracy_sc))
    # print('Recall: {:.3f}'.format(recall_sc))
    # print('Precision: {:.3f}'.format(precision_sc))

    return accuracy_sc, recall_sc, precision_sc

answer_three()

# ### Question 4
#
# Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`,
# what is the confusion matrix when using a threshold of -220 on the decision function.
# Use X_test and y_test.
#
# *This function should return a confusion matrix, a 2x2 numpy array with 4 integers.*

# In[ ]:

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    # Fit
    svm = SVC(C=1e9, gamma=1e-07).fit(X_train, y_train)

    # Decision function returns values for y_pred iso labels
    # Threshold; all ABOVE > threshold is True(fraudulent)
    # Higher TH reduces FP, increases FN, Lower TH increases FP, reduces FN
    fraud_threshold = -220
    y_decision_scores = svm.decision_function(X_test) > fraud_threshold

    # Sanity check threshold
    decision_scores = [(t, s, b) for t, s, b in zip(y_test, svm.decision_function(X_test), y_decision_scores) if t == 1]
    print(np.max(decision_scores, axis=0))  # higher decision score == true == class 1
    print(np.min(decision_scores, axis=0))  # lower decision score == false == class 0

    # Confusion matrix based on threshold
    cm = confusion_matrix(y_test, y_decision_scores.astype('i'))

    # Sanity check 2
    # print(cm)
    # print(list(zip(y_test[0:20], y_decision_scores[0:20].astype('i'))))
    assert cm.shape == (2, 2)
    assert cm.dtype == 'int64'

    return cm

answer_four()


# ### Question 5
#
# Train a logisitic regression classifier with default parameters using X_train and y_train.
#
# For the logisitic regression classifier, create a precision recall curve and a roc curve
# using y_test and the probability estimates for X_test (probability it is fraud).
#
# Looking at the precision recall curve, what is the recall when the precision is `0.75`?
#
# Looking at the roc curve, what is the true positive rate when the false positive rate is `0.16`?
#
# *This function should return a tuple with two floats, i.e. `(recall, true positive rate)`.*

# In[ ]:

def answer_five():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc

    # Fit lr, return probabilities
    lr = LogisticRegression().fit(X_train, y_train)
    y_proba = lr.predict_proba(X_test)[: ,1]

    #
    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])

    # PR curve
    # precision, recall, _ = precision_recall_curve(y_test, y_proba)
    # plt.plot(precision, recall, label='Precision-Recall Curve')
    # plt.ylabel('Recall', fontsize=16)
    # plt.xlabel('Precision', fontsize=16)

    # ROC curve
    false_positive_rate, recall, _ = roc_curve(y_test, y_proba)
    plt.plot(false_positive_rate, recall, label='ROC Curve')
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)

    plt.axes().set_aspect('equal')
    plt.show()

    return 0.825, 0.950


# ### Question 6
#
# Perform a grid search over the parameters listed below for a Logisitic Regression classifier,
# using recall for scoring and the default 3-fold cross validation.
#
# `'penalty': ['l1', 'l2']`
#
# `'C':[0.01, 0.1, 1, 10, 100]`
#
# From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.
#
# |      	| `l1` 	| `l2` 	|
# |:----:	|----	|----	|
# | **`0.01`** 	|    ?	|   ? 	|
# | **`0.1`**  	|    ?	|   ? 	|
# | **`1`**    	|    ?	|   ? 	|
# | **`10`**   	|    ?	|   ? 	|
# | **`100`**   	|    ?	|   ? 	|
#
# <br>
#
# *This function should return a 5 by 2 numpy array with 10 floats.*
#
# *Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array.
# You might need to reshape your raw result to meet the format we are looking for.*

# In[ ]:

def answer_six():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    # Fit grid
    grid_values = {'C':[0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    lr = LogisticRegression(random_state=0)
    grid_lr_prec = GridSearchCV(lr, param_grid=grid_values, scoring='recall', cv=3, return_train_score=False)
    grid_lr_prec.fit(X_train, y_train)

    # Mean test scores of each parameter combination
    df = pd.DataFrame(grid_lr_prec.cv_results_)
    pivot = pd.pivot_table(df, values='mean_test_score', index=['param_C'], columns = ['param_penalty']).as_matrix()

    return pivot

answer_six()

# In[ ]:

# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    import seaborn as sns
    plt.figure()
    sns.heatmap(scores.reshape(5, 2), xticklabels=['L1', 'L2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0)
    plt.axes().set_aspect('equal')

GridSearch_Heatmap(answer_six())



######### notes
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

dataset = load_digits()
X, y = dataset.data, dataset.target

y_binary_imbalanced = y.copy()
y_binary_imbalanced[y_binary_imbalanced != 1] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

m = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
m.fit(X_train, y_train)

m_predicted = m.predict(X_test)
confusion = confusion_matrix(y_test, m_predicted)

print('Logistic regression classifier (default settings)\n', confusion)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, m_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, m_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, m_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, m_predicted)))

from sklearn.metrics import classification_report
print(classification_report(y_test, m_predicted, target_names=['not 1', '1']))

from sklearn.metrics import precision_recall_curve
y_scores_m = m.fit(X_train, y_train).decision_function(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores_m)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]

plt.figure()
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.plot(precision, recall, label='Precision-Recall Curve')
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle='none', c='r', mew=3)
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.axes().set_aspect('equal')
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test, m_predicted, average = 'micro')))

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score

grid_values = {'gamma': [0.01, 0.1, 1, 10]}

# alternative metric to optimize over grid parameters: AUC
m2 = LogisticRegression()
grid_m = GridSearchCV(m2, param_grid = grid_values, scoring = 'precision')
grid_m.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m.decision_function(X_test)

print('Test set Precision: ', precision_score(y_test, y_decision_fn_scores_m))
print('Grid best parameter (max. Precision): ', grid_m.best_params_)
print('Grid best score (Precision): ', grid_m.best_score_)

grid_values = {'gamma': [10]}
grid_m = GridSearchCV(m, param_grid = grid_values, scoring = 'precision')
grid_m.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m.decision_function(X_test)

print('Grid best parameter (max. Precision): ', grid_m.best_params_)
print('Grid best score (Precision): ', grid_m.best_score_)

grid_m = GridSearchCV(m, param_grid = grid_values, scoring = 'recall')
grid_m.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m.decision_function(X_test)

print('Grid best parameter (max. Recall): ', grid_m.best_params_)
print('Grid best score (Recall): ', grid_m.best_score_)


###5
y_scores_m = m.fit(X_train, y_train).decision_function(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores_m)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]

plt.figure()
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.plot(precision, recall, label='Precision-Recall Curve')
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle='none', c='r', mew=3)
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.axes().set_aspect('equal')
plt.show()

###6


###13
grid_values = {'gamma':  [0.01, 0.1, 1, 10]}
grid_m = GridSearchCV(m, param_grid = grid_values, scoring = 'precision')
grid_m.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m.decision_function(X_test)

print('Grid best parameter (max. Precision): ', grid_m.best_params_)
print('Grid best score (Precision): ', grid_m.best_score_)

grid_m = GridSearchCV(m, param_grid = grid_values, scoring = 'recall')
grid_m.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m.decision_function(X_test)

print('Grid best parameter (max. Recall): ', grid_m.best_params_)
print('Grid best score (Recall): ', grid_m.best_score_)


###14
grid_values = {'gamma': [1]}
grid_m = GridSearchCV(m, param_grid = grid_values, scoring = 'precision')
grid_m.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m.decision_function(X_test)

print('Grid best parameter (max. Precision): ', grid_m.best_params_)
print('Grid best score (Precision): ', grid_m.best_score_)

grid_m = GridSearchCV(m, param_grid = grid_values, scoring = 'recall')
grid_m.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m.decision_function(X_test)

print('Grid best parameter (max. Recall): ', grid_m.best_params_)
print('Grid best score (Recall): ', grid_m.best_score_)


###13 2
grid_values = {'gamma': [10]}
grid_m = GridSearchCV(m, param_grid = grid_values, scoring = 'precision')
grid_m.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m.decision_function(X_test)

print('Grid best parameter (max. Precision): ', grid_m.best_params_)
print('Grid best score (Precision): ', grid_m.best_score_)

grid_values = {'gamma': [0.01, 0.1, 1, 10]}

grid_m_recall = GridSearchCV(m, param_grid = grid_values, scoring = 'recall')
grid_m_recall.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m.decision_function(X_test)

print('Grid best parameter (max. Recall): ', grid_m_recall.best_params_)
print('Grid best score (Recall): ', grid_m_recall.best_score_)

grid_m_recall.best_score_ - grid_m.best_score_


###14 2
grid_values = {'gamma': [0.01, 0.1, 1, 10]}
grid_m = GridSearchCV(m, param_grid = grid_values, scoring = 'precision')
grid_m.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m.decision_function(X_test)

print('Grid best parameter (max. Precision): ', grid_m.best_params_)
print('Grid best score (Precision): ', grid_m.best_score_)

grid_values = {'gamma': [1]}

grid_m_recall = GridSearchCV(m, param_grid = grid_values, scoring = 'recall')
grid_m_recall.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m.decision_function(X_test)

print('Grid best parameter (max. Recall): ', grid_m_recall.best_params_)
print('Grid best score (Recall): ', grid_m_recall.best_score_)

grid_m.best_score_ - grid_m_recall.best_score_


###
grid_values = {'C': [0.01, 0.1, 1, 10], 'gamma':  [0.01, 0.1, 1, 10]}
grid_m_prec = GridSearchCV(m, param_grid = grid_values, scoring = 'precision')
grid_m_prec.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m_prec.decision_function(X_test)
print('-'*20)
print('Grid best parameter (max. Precision): ', grid_m_prec.best_params_)
print('Grid best score (Precision): ', grid_m_prec.best_score_)
print('-'*20)
grid_values = {'C': [0.01, 0.1, 1, 10], 'gamma':  [0.01, 0.1, 1, 10]}

grid_m_recall = GridSearchCV(m, param_grid = grid_values, scoring = 'recall')
grid_m_recall.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m_recall.decision_function(X_test)

print('Grid best parameter (max. Recall): ', grid_m_recall.best_params_)
print('Grid best score (Recall): ', grid_m_recall.best_score_)
print('-'*20)
print('Recall - Precision: ', grid_m_recall.best_score_ - grid_m_prec.best_score_)
print('-'*20)

###
# grid_values = {'C': [0.01, 0.1, 1, 10], 'gamma':  [0.01, 0.1, 1, 10]}
grid_values = {'C': [0.01], 'gamma': [0.01]}
grid_m_prec = GridSearchCV(m, param_grid = grid_values, scoring = 'precision')
grid_m_prec.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m_prec.decision_function(X_test)
print('-'*20)
print('Grid best parameter (max. Precision): ', grid_m_prec.best_params_)
print('Grid best score (Precision): ', grid_m_prec.best_score_)
print('-'*20)
grid_values = {'C': [0.01, 0.1, 1, 10], 'gamma':  [0.01, 0.1, 1, 10]}

grid_m_recall = GridSearchCV(m, param_grid = grid_values, scoring = 'recall')
grid_m_recall.fit(X_train, y_train)
y_decision_fn_scores_m = grid_m_recall.decision_function(X_test)

print('Grid best parameter (max. Recall): ', grid_m_recall.best_params_)
print('Grid best score (Recall): ', grid_m_recall.best_score_)
print('-'*20)
print('Recall - Precision: ', grid_m_recall.best_score_ - grid_m_prec.best_score_)
print('-'*20)