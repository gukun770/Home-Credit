import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import set_matplotlib_formats
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

set_matplotlib_formats('retina')
plt.style.use('ggplot')

def shuffle_column(X, feature_index):
    '''
    Parameters
    ----------
    X: numpy array
    feature_index: int

    Returns
    -------
    X_new: numpy array

    Returns a new array identical to X but
    with all the values in column feature_index
    shuffled
    '''

    X_new = X.copy()
    np.random.shuffle(X_new[:,feature_index])
    return X_new

def permutation_importance(model, X_test, y_test, scorer=accuracy_score):
    ''' Calculates permutation feature importance for a fitted model

    Parameters
    ----------
    model: anything with a predict() method
    X_test, y_test: numpy arrays of data
        unseen by model
    scorer: function. Should be a "higher is better" scoring function,
        meaning that if you want to use an error metric, you should
        multiply it by -1 first.
        ex: >> neg_mse = lambda y1, y2: -mean_squared_error(y1, y2)
            >> permutation_importance(mod, X, y, scorer=neg_mse)

    Returns
    -------
    feat_importances: numpy array of permutation importance
        for each feature

    '''

    feat_importances = np.zeros(X_test.shape[1])
    test_score = scorer(y_test, model.predict(X_test))
    for i in range(X_test.shape[1]):
        X_test_shuffled = shuffle_column(X_test, i)
        test_score_permuted = scorer(y_test, model.predict(X_test_shuffled))
        feat_importances[i] = test_score - test_score_permuted
    return feat_importances

def permutation_importance_auc(model, X_test, y_test):
    ''' Calculates permutation feature importance for a fitted classifier model using auc
    '''
    def scorer(y_test, y_pred_prob):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        return auc(fpr, tpr)

    feat_importances = np.zeros(X_test.shape[1])
    test_score = scorer(y_test, model.predict_proba(X_test)[:,1:])
    for i in range(X_test.shape[1]):
        X_test_shuffled = shuffle_column(X_test, i)
        test_score_permuted = scorer(y_test, model.predict_proba(X_test_shuffled)[:,1:])
        feat_importances[i] = test_score - test_score_permuted
    return feat_importances


def replace_column(X, feature_index, value):
    '''
    Parameters
    ----------
    X: numpy array
    feature_index: int
    value: float

    Returns
    -------
    X_new: numpy array
    Returns a new array identical to X but
    with all the values in column feature_index
    replaced with value
    '''
    X_new = X.copy()
    X_new[:,feature_index] = value
    return X_new

def partial_dependence(model, X, feature_index, classification=True):
    '''
    Parameters
    ----------
    model: fitted model
        anything with .predict()
    X: numpy array
        data the model was trained on.
    feature_index: int
        feature to calculate partial dependence for
    classification: boolean.
        True if the model is a classifier
           (in which case, it must have .predict_proba()
        False if the model is a regressor

    Returns
    -------
    x_values: numpy array
        x values to plot partial dependence over
    pdp: numpy array
        partial dependence values

    example:
    >> x, pdp = partial_dependece(model, X_train, 3, classification=False)
    >> plt.plot(x, pdp)
    '''

    x_values = np.unique(X[:,feature_index])
    pdp = np.zeros(x_values.shape)
    for i, value in enumerate(x_values):
        X_new = replace_column(X, feature_index, value)
        if classification:
            y_pred_prob = model.predict_proba(X_new)[:,1]
            y_pred_prob = np.clip(y_pred_prob, 0.001, 0.999)
            y_pred = np.log(y_pred_prob / (1 - y_pred_prob))
        else:
            y_pred = model.predict(X_new)
        pdp[i] = y_pred.mean()
    return (x_values, pdp)


def plot_feature_importance(importance_series, show_n=None):

    if show_n:
        plt.figure(figsize=(10,10))
        importance_series.sort_values()[-1*show_n:].plot(kind='barh', width=0.35)
    else:
        plt.figure(figsize=(12,14))
        importance_series.sort_values().plot(kind='barh')
    plt.title('Permutational Feature Importance')
    plt.tight_layout()