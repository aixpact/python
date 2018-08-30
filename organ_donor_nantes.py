#
# Title: Management of the heartbeating brain-dead organ donor - Organ donor management for BD organ donors
#
# Summary
# The main factor limiting organ donation is the availability of suitable donors and organs.
# Currently, most transplants follow multiple organ retrieval from heartbeating brain-dead organ donors.
# However, brain death is often associated with marked physiological instability, which, if not managed,
# can lead to deterioration in organ function before retrieval. In some cases, this prevents successful donation.
# There is increasing evidence that moderation of these pathophysiological changes by active management in Intensive Care
# maintains organ function, thereby increasing the number and functional quality of organs available for transplantation.
# This strategy of active donor management requires an alteration of philosophy and therapy on the part of the
# intensive care unit clinicians and has significant resource implications if it is to be delivered reliably and safely.
# Despite increasing consensus over donor management protocols, many of their components have not yet been subjected to controlled evaluation.
# Hence the optimal combinations of treatment goals, monitoring, and specific therapies have not yet been fully defined.
# More research into the component techniques is needed.
#
# Objective:
# Optimize the number of good functioning organs suitable for donation by good organ reanimation/perfusion.
#
# Data:
# Retrospective data of donors containing features and targets as per excel file.
#
#
# Analyse the major risk factor for each organ in order to optimize the number of organs grafted.
#
# Analysis stages:
# 1. analyse the total population == infer patterns among all donors to identify the major risk factor(s)
# 2. compare population; analyse/identify risk factors - Compare donors who gave 2 or less organs vs. 3 or more
# 3. analyse/identify analyse risk factor for each organ given or not
# We tought to use a multiple regression analysis also to build the better model
#
# Client: Paul Rooze (MD), MC Nantes
# Author: Frank J. Ebbers
# Date: 13 feb 2018


# import matplotlib as mpl
# mpl.use('Agg')

# Works in .py
# import matplotlib
# matplotlib.use('TkAgg')


# ------------------------------------------ Import libraries ------------------------------------------#
from collections import Counter
import numpy as np
from operator import itemgetter
import pandas as pd
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# Import preprocessing, selection and metrics
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

# Import classifiers
from sklearn import tree
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC

# ------------------------------------------ General Helper functions ------------------------------------------#

def correlation_matrix(df):
    corr = df.apply(lambda x: x.factorize()[0]).corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    plt.figure()
    with sns.axes_style("white"):
        ax = sns.heatmap(corr, mask=mask, cmap = 'RdBu_r', vmin=-.3, vmax=.3, annot=True, square=True)
        plt.title("Feature Correlations", y=1.03, fontsize=20)
        plt.tight_layout()

def correlation(df, features):
    return df.loc[:, features].apply(lambda x: x.factorize()[0]).corr()

def assert_na(df, feature):
    message = 'Feature {} must be cleaned further'.format(feature)
    assert ~df.loc[:, feature].isnull().values.any(), message
    assert ~df.loc[:, feature].isin(['ND']).values.any(), message

def impute(df, feature, by):
    list_ = df.loc[:, feature].replace([np.nan, None, 'ND', 'nan', 'NaN', 'NA', ''], 9999)
    mask_ = list_==9999
    if by == 'mean':
        list_[mask_] = list_[~mask_].mean()
    elif by == 'mode':
        list_[mask_] = list_[~mask_].mode()
    elif by in df.columns:
        list_[mask_] = df.loc[mask_, by]
    else:
        list_[mask_] = by
    # print(list_, list_.isnull().values.any())
    # assert list_.isnull().values.any(), feature
    return list_

def clean_seq(df, feature):
    return ['_'.join(sorted(set(re.split(';\s*|\.\s*|,\s*|\+\s*|\s+', str(x).strip()))))
                      for x in df.loc[:, feature]]

def feature_importance(df, y, model):
    plt.figure()
    clf = model.fit(df, y)
    zipr = sorted(zip(df.columns, clf.feature_importances_), key=lambda x: x[1])
    D =  {ftr:score for ftr, score in zipr if score > 0}
    plt.title('Feature Importance Ranking: {}'.format(y.name))
    plt.barh(range(len(D)), list(D.values()), align='center')
    plt.yticks(range(len(D)), list(D.keys()))
    plt.tight_layout()
    plt.show()
    return D

def cv_scores(model, *args, k=5):
    X, y = args
    cv_scores = cross_val_score(model, X, y, cv=k, n_jobs=-1)
    print('Cross-validation Accuracies ({}-folds): {}\nMean Accuracy: {}'.format(k,
                            np.round_(cv_scores, 3), np.round_(np.mean(cv_scores), 3)))
    return None

def auc_scores(model, *args, k=10, threshold=0.50):
    """CV scores"""
    X, y = args
    predictions = cross_val_predict(model, X, y, cv=k, n_jobs=-1)
    print('AUC - Test predict  {:.2%}'.format(roc_auc_score(y, predictions)))
    try:
        pred_probas = (cross_val_predict(model, X, y, cv=k, method='predict_proba', n_jobs=-1)[:, 1] > threshold) * 1
        print('AUC - Test probabil {:.2%}'.format(roc_auc_score(y, pred_probas)))
    except:
        pass
    return None

def show_tree(clf, features, labels):
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=features, class_names=labels,
                             filled=True, rounded=True, special_characters=True, impurity=True, proportion=False)
    graphviz.Source(dot_data).view()

def show_distribution(df, feature, plot_out=False):
    distr = df.loc[:, feature].value_counts()
    if plot_out:
        plt.figure()
        sns.set()
        try:
            ax = sns.distplot(distr)
            plt.tight_layout()
            plt.show()
        except:
            pass
    return distr

def show_frequency(df, feature, plot_out=False):
    top = 15
    clrs = ['cornflowerblue', 'royalblue', 'navy']
    lbls = ['2- organs grafted','3+ organs grafted']
    bars = len(df.loc[:, feature].value_counts().nlargest(top))
    if plot_out:
        plt.figure(figsize=(12, 8))
        sns.set()
        M = df['sum_grafted']<3
        ax = df.loc[M, feature].value_counts().nlargest(top).plot(kind='barh', figsize=(12, 8), color=clrs[0],
                                                                  label=lbls[0], fontsize=12, alpha=0.75)
        df.loc[~M, feature].value_counts().nlargest(top).plot(kind='barh', figsize=(12, 8), color=clrs[1],
                                                              label=lbls[1], fontsize=12, alpha=0.75, ax=ax)
        ax.legend(loc=4)
        ax.set_alpha(0.8)
        ax.set_title('Frequency of {}'.format(feature), fontsize=14)
        ax.set_xlabel('Count', fontsize=12)
        ax.set_xlim(right=max(df.loc[M, feature].value_counts())*1.2)
        total_width = sum([i.get_width() for i in ax.patches])
        # get_width pulls left or right; get_y pushes up or down
        for i, x in enumerate(ax.patches):
            annot = '{0}({1:.0f}%)'.format(x.get_width(), x.get_width()/total_width*100)
            x_annot = x.get_width()+x.get_width()/40
            y_annot = x.get_y()+0.05/bars+0.45
            ax.text(x_annot, y_annot, annot, color=clrs[2], fontsize=int(max(6, min(14, 40/bars))))
        ax.invert_yaxis() # sort barlenghth long at top
        plt.tight_layout()
        plt.show()
    return None

def show_correlation(df, feature, target, plot_out=False):
    if plot_out:
        plt.figure(figsize=(8, 8))
        try:
            g = sns.jointplot(x=df.loc[:, feature], y=df.loc[:, target], size=10, kind='reg')
            plt.tight_layout()
            plt.show()
        except:
            plt.close()
            factors = df.loc[:, feature].factorize()
            labels = factors[1].tolist()
            g = sns.jointplot(x=factors[0], y=df.loc[:, target], size=10, kind='reg') #, annot_kws=dict(stat="r"))
            g.ax_joint.set_xticks(range(0, len(labels)))
            g.ax_joint.set_xticklabels(labels)
            g.ax_joint.set_xlabel(feature)
            plt.tight_layout()
            plt.show()
    return None

# ---------------------------------------------> Import data <-------------------------------------------------------- #
# Import data
df = pd. read_excel('pmo_altered.xlsx', skiprows=2)
df.head()
df.info()

# ---------------------------------------------> Feature set cleaning <--------------------------------------------------- #

# Collect features that do not contribute to inference and/or prediction
garbage_features = []

# Formula features:
# => have 1.0 correlation with other features
garbage_features.extend(['bmi_formula', 'norm_weight_formula', 'weight_diff_formula',
         'mdrd_clairanc_formula', 'resp_tidal_volume_formula',
         'balance_input_output_formula'])

# Features: potential_organs, bilan_no
# Human pre-prediction contain information about/correlate with target
garbage_features.extend(['potential_organs', 'bilan_no'])

# Feature: ipp
# Internal admin data only
garbage_features.append('ipp')

# Feature: sexe
# Can't infer NA's; drop feature from df
garbage_features.append('sexe')

# Drop all 'garbage' features
df.drop(garbage_features, axis=1, inplace=True)

# ---------------------------------------------> Exploratory Data Analysis <-------------------------------------------- #

# Set constants for EDA
show_plots = True
cor_target = 'sum_grafted'


# ---------------------------------------------> Feature wrangling <-------------------------------------------- #

# TODO Infer NA's of sexe



# Feature: height                        213 non-null object
# Replace NA by average height (disregarding sexe)
# Correlation with number of grafted organs: 0.004694
feature = 'height'
# show_distribution(df, feature, show_plots)
df.loc[:, feature] = impute(df, feature, 'mean').astype('f')
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, feature, cor_target, show_plots)
assert_na(df, feature)

# Feature: paraclinical_confirmation     213 non-null object
# Change typos to mode (peak value)
# Correlation with number of grafted organs: -0.054312
feature = 'paraclinical_confirmation'
# show_distribution(df, feature, show_plots)
df.loc[:, feature] = np.where(df.loc[:, feature] == 2, 2, df.loc[:, feature].mode())
df.loc[:, feature] = df.loc[:, feature].astype('i')
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, feature, cor_target, show_plots)
assert_na(df, feature)

# Feature: antibiotic_before_surgery     213 non-null object
# high number of NA's => change NA's to 2 and make categorical dummy vars
# Correlation with number of grafted organs: -0.069872
feature = 'antibiotic_before_surgery'
# show_distribution(df, feature, show_plots)
df.loc[:, feature] = impute(df, feature, 2)
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, feature, cor_target, show_plots)
assert_na(df, feature)
df = pd.get_dummies(df, columns=[feature], drop_first=True)

# Feature: antibiotic_at_surgery         142 non-null object
# high number of NA's => change NA's to 2 and make categorical dummy vars
# Correlation with number of grafted organs: -0.052961
feature = 'antibiotic_at_surgery'
# show_distribution(df, feature, show_plots)
df.loc[:, feature] = impute(df, feature, 2)
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, feature, cor_target, show_plots)
assert_na(df, feature)
df = pd.get_dummies(df, columns=[feature], drop_first=True)

# Feature: days_atb_treatment            124 non-null object
# high number of NA's => change NA's to 0 and make categorical dummy vars
# Correlation with number of grafted organs: 0.119873 **
feature = 'days_atb_treatment'
# show_distribution(df, feature, show_plots)
df.loc[:, feature] = impute(df, feature, 0)
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, feature, cor_target, show_plots)
assert_na(df, feature)
df = pd.get_dummies(df, columns=[feature], drop_first=True)

# Feature: infection_local               122 non-null object
# high number of NA's => change NA's to 9 and make categorical dummy vars
# Correlation with number of grafted organs: 0.149374 **
feature = 'infection_local'
# show_distribution(df, feature, show_plots)
df.loc[:, feature] = impute(df, feature, 9)
df.loc[:, feature] = clean_seq(df, feature)
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, feature, cor_target, show_plots)
assert_na(df, feature)
df = pd.get_dummies(df, columns=[feature], drop_first=True)

# Feature: infection_type                116 non-null object
# high number of NA's => change NA's to 9 and make categorical dummy vars
# Correlation with number of grafted organs: 0.130116 **
# TODO Uniform vector seperators
feature = 'infection_type'
# show_distribution(df, feature, show_plots)
df.loc[:, feature] = impute(df, feature, 9)
df.loc[:, feature] = clean_seq(df, feature)
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, feature, cor_target, show_plots)
assert_na(df, feature)
df = pd.get_dummies(df, columns=[feature], drop_first=True)

# Feature: last_weight_before_surgery    212 non-null object
# Impute NA's by value of weight (assume no weight loss)
# Correlation with number of grafted organs: 0.113497 **
feature = 'last_weight_before_surgery'
show_distribution(df, feature, show_plots)
df.loc[:, feature] = impute(df, feature, 'weight')
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, feature, cor_target, show_plots)
assert_na(df, feature)

# Feature: nad_max                       213 non-null object
# impute NA's ('Manque poids') by mean
# Correlation with number of grafted organs: 0.0943 *
feature = 'nad_max'
# show_distribution(df, feature, show_plots)
df.loc[:, feature] = df.loc[:, feature][df[feature].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
df.loc[:, feature] = df.loc[:, feature].astype('f')
df.loc[:, feature] = impute(df, feature, 'mean')
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, feature, cor_target, show_plots)
assert_na(df, feature)

# Feature: hormones_thyroïdiennes        213 non-null object
# 211/213 values are equal, feature can be dropped
feature = 'hormones_thyroïdiennes'
show_distribution(df, feature, show_plots)
df.drop(feature, axis=1, inplace=True)

# Feature: monitorage                    207 non-null object
# Impute NA's by 9 and make categorical dummy vars
# Correlation with number of grafted organs: -0.093116 *
feature = 'monitorage'
# show_distribution(df, feature, show_plots)
df.loc[:, feature] = clean_seq(df, feature)
df.loc[:, feature] = impute(df, feature, '9')
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, feature, cor_target, show_plots)
assert_na(df, feature)
df = pd.get_dummies(df, columns=[feature], drop_first=True)

# Feature: hours_proceedings             213 non-null object
# Impute NA by mean
# Correlation with number of grafted organs: -0.004165
feature = 'hours_proceedings'
df.loc[:, feature] = impute(df, feature, 'mean').astype('f')
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, feature, cor_target, show_plots)
show_distribution(df, feature, show_plots)
assert_na(df, feature)

# Feature: hema_transfusion              212 non-null float32
# Impute by mode
# Correlation with number of grafted organs:
feature = 'hema_transfusion'
df.loc[pd.isnull(df.loc[:, feature]), feature] = 99
df.loc[:, feature].fillna(99, inplace=True)
df.loc[df.loc[:, feature]==99, feature] = df.loc[~pd.isnull(df.loc[:, feature]), feature].mode()
df.loc[:, feature].fillna(99, inplace=True)
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, feature, cor_target, show_plots)
show_distribution(df, feature, show_plots)
assert_na(df, feature)

# Feature: loxen                           212 non-null float64
# One NA replace by 0 (majority)
# Correlation with number of grafted organs:
feature = 'loxen'
show_distribution(df, feature, show_plots)
df.loc[:, feature] = impute(df, feature, 0).astype(np.uint8)
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, feature, cor_target, show_plots)
assert_na(df, feature)

# ---------------------------------------------> Target wrangling <-------------------------------------------- #

# Target: organs_taken                  213 non-null object
# Clean seperators
# Correlation with number of grafted organs: 0.417287 ***
# Make categorical and create dummy vars
feature = 'organs_taken'
# show_distribution(df, feature, show_plots)
df.loc[:, feature] = clean_seq(df, feature)
show_frequency(df, feature, show_plots)
correlation(df, [feature, cor_target])
show_correlation(df, cor_target, feature, show_plots)
assert_na(df, feature)
df = pd.get_dummies(df, columns=[feature], drop_first=True)

# ---------------------------------------------> TODO plot all features? <-------------------------------------------- #

for feature in df.columns[5:10]:
    show_frequency(df, feature, True)


# ---------------------------------------------> Final Feature check <-------------------------------------------- #

# Final check for Inf's, NA's, ND's
df.replace([np.inf, -np.inf], np.nan)
for ftr in df.columns:
    assert ~df.loc[:, ftr].isnull().values.any(), df.loc[:, ftr].isnull().values.sum()
    assert ~df.isin(['ND']).values.any(), 'Feature {} still contains ND'.format(ftr)


# ---------------------------------------------> Set features types <-------------------------------------------- #

# Remove single value features
singleton_features = [feature for i, feature in enumerate(df.columns)
                    if len(df.loc[:, feature].value_counts())<2]
df.drop(singleton_features, axis=1, inplace=True)

# Convert bivariate features to booleans
for feature in df.columns:
    if len(df.loc[:, feature].value_counts()) == 2:
        try:
            df.loc[:, feature] = df.loc[:, feature].astype(np.uint8)
        except:
            print("{} can't be converted to boolean".format(feature))
            continue

# Convert multivalue features to floats
for feature in df.columns:
    if len(df.loc[:, feature].value_counts()) > 2:
        try:
            df.loc[:, feature] = df.loc[:, feature].astype('f')
        except:
            print("{} can't be converted to float".format(feature))
            continue

# ---------------------------------------------> Prepare holdout set <-------------------------------------------- #

# Stratified holdout test sampling, for prediction verification:
# ±10% test donors = ±21 donors ± 2 donors per center
df_holdout = df.groupby('center', group_keys=False).apply(lambda x: x.sample(min(len(x), 2)))
# df_holdout.head()
# df_holdout.info()

holdout_index = df_holdout.index.tolist()
df['holdout'] = 0
df.loc[holdout_index, 'holdout'] = 1

# ---------------------------------------------> Targets <-------------------------------------------- #

organs = []
for column in df.columns:
    organs.extend(re.findall(r'organs_taken_.*', column))

targets = ['kidney_grafted', 'heart_grafted', 'lung_grafted', 'liver_grafted',
                  'pancreas_grafted', 'sum_grafted', 'no_of_organs_taken']

targets.extend(organs)
df_tgt = df.loc[:, targets]
df.drop(targets, axis=1, inplace=True)

# Drop identity features
df_donor = df.loc[:, ['donor', 'center']]
df.drop(['donor', 'center'], axis=1, inplace=True)

# Mask holdout
model_features = set(df.columns) ^ set(['holdout'])

# ---------------------------------------------> Scale floats <-------------------------------------------- #

floats = df.select_dtypes('f').columns
M = df['holdout']==0

scaler = MinMaxScaler()
df.loc[M, floats] = scaler.fit_transform(df.loc[M, floats])
df.loc[~M, floats] = scaler.transform(df.loc[~M, floats])

# ---------------------------------------------> Cross Validation <-------------------------------------------- #

model = DecisionTreeClassifier(max_depth=5, max_features=15, random_state=0)
feature_importance(df.loc[M, model_features], df_tgt.loc[M, 'heart_grafted'], model)

# ---------------------------------------------> Model tuning and selection <----------------------------------------- #

# Due to small sample size, overfitting is a real concern
# Models need to be regularized (fewer parameters/lower weights) to generalize well
classifiers1 = [
    DummyClassifier(strategy='most_frequent', random_state=0),
    DummyClassifier(strategy='stratified', random_state=0),
    GaussianNB(),
    LogisticRegression(C=0.5, random_state=0, penalty='l1'),
    LogisticRegression(C=0.5, random_state=0, penalty='l2'),
    LogisticRegression(C=1.0, random_state=0, penalty='l1'),
    LogisticRegression(C=1.0, random_state=0, penalty='l2'),
    LogisticRegression(C=5.0, random_state=0, penalty='l1'),
    LogisticRegression(C=5.0, random_state=0, penalty='l2'),
    LogisticRegression(C=10.0, random_state=0, penalty='l1'),
    LogisticRegression(C=10.0, random_state=0, penalty='l2'),
    DecisionTreeClassifier(max_depth=3, max_features=10, random_state=0),
    DecisionTreeClassifier(max_depth=4, max_features=15, random_state=0),
    DecisionTreeClassifier(max_depth=5, max_features=20, random_state=0),
    DecisionTreeClassifier(max_depth=5, max_features=None, random_state=0),
    DecisionTreeClassifier(max_depth=6, max_features=None, random_state=0),
    DecisionTreeClassifier(max_depth=7, max_features=None, random_state=0),
    DecisionTreeClassifier(max_depth=8, max_features=None, random_state=0),
    DecisionTreeClassifier(max_depth=9, max_features=None, random_state=0),
    GradientBoostingClassifier(random_state=0),
    GradientBoostingClassifier(learning_rate=0.08, max_depth=3, max_features=10, random_state=0),
    GradientBoostingClassifier(learning_rate=0.12, max_depth=4, max_features=10, random_state=0),
    GradientBoostingClassifier(learning_rate=0.12, max_depth=5, max_features=10, random_state=0),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=4, max_features=None, random_state=0),
    RandomForestClassifier(n_estimators=100, max_leaf_nodes=4, random_state=0),
    RandomForestClassifier(n_estimators=100, max_leaf_nodes=None, random_state=0),
    AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=0),
    KNeighborsClassifier(n_neighbors=5),
    LinearSVC(random_state=0, C=1.0),
    LinearSVC(random_state=0, C=5.0),
    LinearSVC(random_state=0, C=10.0),
    SVC(C=1.0, random_state=0),
    SVC(C=5.0, random_state=0),
    SVC(C=10.0, random_state=0),
]

classifiers2 = [
    LogisticRegression(C=10.0, random_state=0, penalty='l1'),
    LogisticRegression(C=20.0, random_state=0, penalty='l1'),
    LogisticRegression(C=40.0, random_state=0, penalty='l1'),
    LogisticRegression(C=60.0, random_state=0, penalty='l1'),
    DecisionTreeClassifier(max_depth=5, max_features=None, random_state=0),
    DecisionTreeClassifier(max_depth=5, max_features=40, random_state=0),
    DecisionTreeClassifier(max_depth=5, max_features=20, random_state=0),
    DecisionTreeClassifier(max_depth=5, max_features=15, random_state=0),
    DecisionTreeClassifier(max_depth=5, max_features=10, random_state=0),
    DecisionTreeClassifier(max_depth=5, max_features=5, random_state=0),
    LinearSVC(random_state=0, C=0.5),
    LinearSVC(random_state=0, C=1.0),
]

classifiers3 = [
    DecisionTreeClassifier(max_depth=5, max_features=15, random_state=0),
    DecisionTreeClassifier(max_depth=5, max_features=20, random_state=0),
    DecisionTreeClassifier(max_depth=5, max_features=25, random_state=0),
    DecisionTreeClassifier(max_depth=5, max_features=30, random_state=0),
]

# CV scores and feature importance for classifiers
# Binary classifications
for organ in targets[:5]:
    X_model, y_model = df.loc[M, model_features], df_tgt.loc[M, organ]
    for model in classifiers3:
        print('-'*80)
        print('Reults with respect to {}:\n'.format(organ))
        print('Model details:\n{}'.format(model))
        # Training scores
        clf = model.fit(X_model, y_model)
        pred_model = clf.predict(X_model)
        print('Scores:\nAUC - Train pred    {:.2%}'.format(roc_auc_score(y_model, pred_model)))
        # CV scores
        clf = model.fit(X_model, y_model)
        auc_scores(clf, X_model, y_model)
        try:
            print('\n')
            print('Feature importance:\{}'.format(feature_importance(df.loc[M, model_features], y_model, model)))
        except:
            print('NA')
            continue


# Scaled Tree
clf = tree.DecisionTreeClassifier(max_depth=5, max_features=20, random_state=0).fit(df.loc[M, model_features], df_tgt.loc[M, 'heart_grafted'])
show_tree(clf, list(model_features), ['not grafted', 'grafted'])

# Unscaled tree
df.loc[M, floats] = scaler.inverse_transform(df.loc[M, floats]) # NOTE: each time scaler scales values!
clf = tree.DecisionTreeClassifier(max_depth=4, max_features=20, random_state=0).fit(df.loc[M, model_features], df_tgt.loc[M, 'heart_grafted'])
show_tree(clf, list(model_features), ['not grafted', 'grafted'])

# ---------------------------------------------> Multilabel <-------------------------------------------- #

multilabel = [
    DecisionTreeClassifier(max_depth=5, max_features=20, random_state=0),
    ExtraTreeClassifier(max_depth=5, max_features=20, random_state=0),
    # RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False)
]

# CV scores and feature importance for classifiers
# Binary classifications
# for organ in targets[5]:
X_model, y_model = df.loc[M, model_features], df_tgt.loc[M, targets[:5]]
for model in multilabel:
    print('-'*80)
    print('Reults with respect to {}:\n'.format(targets[:5]))
    print('Model details:\n{}'.format(model))
    # Training scores
    clf = model.fit(X_model, y_model)
    pred_model = clf.predict(X_model)
    print('Scores:\nAUC - Train pred    {:.2%}'.format(roc_auc_score(y_model, pred_model)))
    # CV scores
    clf = model.fit(X_model, y_model)
    auc_scores(clf, X_model, y_model)
    try:
        print('\n')
        print('Feature importance:\{}'.format(feature_importance(df.loc[M, model_features], y_model, model)))
    except:
        print('NA')
        continue

# Scaled Tree Plots
clf = tree.DecisionTreeClassifier(max_depth=5, max_features=20, random_state=0).fit(
    df.loc[M, model_features], df_tgt.loc[M, targets[:5]])
show_tree(clf, list(model_features), ['kidney', 'heart', 'lung', 'liver', 'pancreas'])

clf = tree.ExtraTreeClassifier(max_depth=5, max_features=20, random_state=0).fit(
    df.loc[M, model_features], df_tgt.loc[M, targets[:5]])
show_tree(clf, list(model_features), ['kidney', 'heart', 'lung', 'liver', 'pancreas'])

# Reverse scaling
df.loc[M, floats] = scaler.inverse_transform(df.loc[M, floats]) # NOTE: each time scaler scales values!

# Unscaled Tree plots
clf = tree.DecisionTreeClassifier(max_depth=4, max_features=20, random_state=0).fit(
    df.loc[M, model_features], df_tgt.loc[M, targets[:5]])
show_tree(clf, list(model_features), ['kidney', 'heart', 'lung', 'liver', 'pancreas'])

clf = tree.ExtraTreeClassifier(max_depth=5, max_features=20, random_state=0).fit(
    df.loc[M, model_features], df_tgt.loc[M, targets[:5]])
show_tree(clf, list(model_features), ['kidney', 'heart', 'lung', 'liver', 'pancreas'])

# ---------------------------------------------> Fit model <-------------------------------------------- #




# ---------------------------------------------> Prediction <-------------------------------------------- #

# Predict holdout set
predicted = clf.predict(df.loc[~M, model_features])
# predicted = pd.DataFrame(clf.predict_proba(X_holdout), columns=clf.classes_).loc[:, 1]
pred_series = pd.DataFrame(predicted)
pred_series['donor'] = df_donor.loc[~M, 'donor'].values
pred_series.set_index('donor', inplace=True)
pred_series

# clf.coef_
clf.tree_


# feature = 'last_weight_before_surgery'
# high numer of NA's; impute NA weights from gaussian distribution randomly
# mean_weight = df.loc[df.loc[:, feature] != 'ND', feature].mean()
# missing_weights = df.loc[df.loc[:, feature] == 'ND', feature].value_counts()
# gaussian_weights = np.random.normal(mean_weight, 1, missing_weights)
# df.loc[df.loc[:, feature] == 'ND', feature] = gaussian_weights
# df.loc[:, feature] = df.loc[:, feature].astype('f')
# show_distribution(df, feature, show_plots)
# assert_na(df, feature) # TODO further cleaning
# last_weight_before_surgery    212 non-null float32

# organs_taken = ['organs_taken_1', 'organs_taken_1_2',
#        'organs_taken_1_2_3', 'organs_taken_1_2_3_4', 'organs_taken_1_2_3_4_5',
#        'organs_taken_1_2_4', 'organs_taken_1_2_4_5', 'organs_taken_1_3',
#        'organs_taken_1_3_4', 'organs_taken_1_3_4_5', 'organs_taken_1_4',
#        'organs_taken_1_4_5', 'organs_taken_2', 'organs_taken_2_4_5',
#        'organs_taken_3_4', 'organs_taken_4']

# TODO
# AUC curves
# Correlation
# Model info; coef_, intercept_
# Lasso, Ridge classifier
# inverse_scaling in tree
# Multiclass classification - regression on qty of organs




