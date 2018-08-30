# https://www.kaggle.com/c/detroit-blight-ticket-compliance/data

# ------------------------------------------ Import libraries ------------------------------------------#
import numpy as np
import pandas as pd
from scipy.stats import itemfreq
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re

# import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ------------------------------------------ Settings and CONSTANTS ------------------------------------------#

from matplotlib import rcParams
rcParams['figure.figsize'] = 12, 8


# ------------------------------------------ Import Helper libraries ------------------------------------------#

# pass

# ------------------------------------------ General Helper functions ------------------------------------------#

def array2df(X_train, feature_names):
    """Convert np array to df, use with correlation matrix"""
    return pd.DataFrame(X_train, columns=feature_names)

def time_lap(start_time=None):
    """Stopwatch, No param = set, param (start) is time elapsed since start"""
    from time import time, gmtime, strftime
    if start_time == None:
        return time()
    return strftime("%Hh:%Mm:%Ss", gmtime(time() - start_time))

def clean_locals(locals):
    params = ''
    try:
        params = str(list(locals)[0]).replace('\n', '')
        params = re.sub(' +', ' ', params)
        params = re.search(r'\((.*)\)', params).group(1)
    except:
        pass
    return params

def num_features(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def cat_features(df):
    return df.select_dtypes(include=['object', 'category']).columns

def date_features(df):
    return df.columns.str.extractall(r'(.*date.*)')[0].values.tolist()

def clip_outliers(values, p=99):
    """clip at 1.5 IQR"""
    min = np.percentile(values, 100-p)
    max = np.percentile(values, p)
    return np.clip(values, min, max)

def numerize_code(df, feature, replace=0):
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
    nums = df[feature].fillna(replace).astype('int64')
    return clip_outliers(nums)

def alpharize_code(df, feature, bins, replace=0, cut=pd.qcut, upper=True):
    zipcode = numerize_code(df, feature, replace)
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:bins]
    if not upper:
        labels = list(map(lambda x:x.lower(), labels))
    return cut(zipcode, bins, labels=labels)

def factorized(df):
    if isinstance(df, pd.Series):
        return df.factorize()[0]
    return df.loc[:, cat_features(df)].apply(lambda x: x.factorize()[0])

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

def show_hist(df, features):
    c, r, h = features
    g = sns.FacetGrid(df, col=c,  row=r)
    g = g.map(plt.hist, h)

def cat_distribution(df, cat_feature, target):
    df[cat_feature] = df[cat_feature].fillna('missing')
    group = df.groupby([cat_feature, target])[target].count().unstack(level=0).T.fillna(0)
    return group

#------------------------------------------ Import dataset ------------------------------------------#

# Import
set_types = {'compliance':object}
df_train = pd.read_csv('train.csv', encoding='latin-1', low_memory=False, dtype=set_types)
df_unseen = pd.read_csv('test.csv', encoding='latin-1', low_memory=False, dtype=set_types)
df_zip = pd.read_csv('zipcode.csv')


# ------------------------------------------ Glance at and copy dataset ------------------------------------------#

# Explore; Types, NaN's, head
# df_train.info()
# df_unseen.info()

# Generalize working phase
df = df_train.copy(deep=True)
flag_unseen_data = False
# df.head()
# df.info()

# ------------------------------------------ Glance at and prepare target ------------------------------------------#

# Set y(target) to proper category type
try:
    df['compliance'] = df['compliance'].fillna(2).replace({'0.0': 0, '1.0': 1}).astype('category')
except:
    flag_unseen_data = True
    pass # for unseen df
# flag_unseen_data



# ------------------------------------------ Glance at numerical features ------------------------------------------#
# Stats
# df.describe()

# Correlation
if not flag_unseen_data:
    correlation_matrix(df.loc[:, df.describe().columns])


# ------------------------------------------ Glance at categorical features ------------------------------------------#

# Factorize features
# factorized(df.iloc[:, 0])

# # Categorical distribution (Feature x target)
# if not flag_unseen_data: TODO error
#     cat_distr_dfs = [cat_distribution(df, feature, 'compliance') for feature in cat_features(df)]
#     cat_distr_dfs[0]

# target_distr = pd.DataFrame(itemfreq(df[target]))
# cat_distr = pd.DataFrame(itemfreq(df[cat_feature]))

# cat_distribution(df, 'country', 'compliance')               # >>>> USA =1, other = 0
# cat_distribution(df, 'city', 'compliance')                  # garbage
# cat_distribution(df, 'state', 'compliance')                 # keep
# cat_distribution(df, 'violator_name', 'compliance')         # garbage
# cat_distribution(df, 'violation_description', 'compliance') # keep?
# cat_distribution(df, 'violation_code', 'compliance')        # keep?
# cat_distribution(df, 'admin_fee', 'compliance')             # keep?

# # Correlations
# correlation(df, ['compliance_detail', 'compliance'])
# correlation(df, ['violation_code', 'violation_description', 'compliance'])
# correlation(df, ['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'disposition', 'collection_status', 'compliance'])
# correlation(df, ['late_fee', 'state_fee', 'admin_fee', 'judgment_amount'])  # admin_fee == state_fee, judgement_amount ± late_fee
# correlation(df, ['hearing_date', 'ticket_issued_date', 'compliance'])

# TODO
# show_hist(df, ('compliance', 'country', 'judgment_amount'))


# ------------------------------------------ Garbage Feature Collection ------------------------------------------#

# Check garbage features
# - which are useless; too contaminated with NaN's
# - which are uniform; one value only
# - which are random, non-generalizing and/or related to instance/index only
# - which are features not present in testset nor production
# - which are directly related to or derived from target (leakage)
# - which are targets, target names

nan_features = ['violation_zip_code', 'non_us_str_code', 'payment_date', 'collection_status', 'grafitti_status']
                # list(df.columns[df.isnull().mean() > 0.50])
zero_ftrs = ['violation_zip_code ']  # df.describe().iloc[0, :] == 0
zero_features = ['non_us_str_code']  # zero_ftrs[zero_ftrs].index.tolist()
uniform_features = ['clean_up_cost']
non_generalizing_features = ['ticket_id', 'violator_name', 'violation_street_number','violation_street_name','violation_zip_code',
                            'mailing_address_str_number','mailing_address_str_name','city']
correlated_features = ['violation_code', 'compliance_detail', 'hearing_date', 'late_fee', 'state_fee']
leakage_features = ['disposition', 'collection_status']
train_only_features = ['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status']
target_features = ['compliance']
garbage_features = (set(nan_features) | set(zero_features) | set(uniform_features) | set(non_generalizing_features) |
                    set(correlated_features) | set(leakage_features) | set(train_only_features))
garbage_features = (set(df.columns) & set(garbage_features))  # correction (unseen df) for garbage features that are not in df
usable_features = (set(df.columns) ^ set(garbage_features))
correlation_matrix(df.loc[:, usable_features])

# EDA Clean and split featureset
df.drop(garbage_features, axis=1, inplace=True)

# ------------------------------------------ Prepare Target and label Features ------------------------------------------#

# Split target feature; y from X
# if not flag_unseen_data:
#     y = df.pop('compliance')
#     y_names = ['non-compliant', 'compliant', 'not responsible']


# ------------------------------------------ Visualize EDA subset ------------------------------------------#

# correlation_matrix(df)


# ------------------------------------------ Feature engineering ------------------------------------------#

# df['wait'] = pd.DataFrame(df['hearing_date'] - df['ticket_issued_date']).apply(np.float32)
# correlation_matrix(df)

# Devide zip codes in 25 locations based on quantiles
# df['zip_code'] = alpharize_code(df_train, 'zip_code', 25, 59999)

# Factorize in levels
df['zip_code'] = factorized(df['zip_code'])
df['agency_name'] = factorized(df['agency_name'])
df['inspector_name'] = factorized(df['inspector_name'])
df['violation_description'] = factorized(df['violation_description'])

# Detroit yes/no
df['state'] = factorized(df['state'])
# df.loc[df['state'] != 0, 'state'] = 1 # TODO tuning

# USA yes/no
# df['USA'] = 0
# df.loc[df['country'] == 'USA', 'country'] = 1
# df.loc[df['country'] != 1, 'country'] = 0
# or same
# df.set_value(df['country'] == 'USA', 'country', 0)
# or same:
df['country'] = np.where(df['country'] == 'USA', 1, 0)
df['country'].mean()


# itemfreq(df['USA'])
# df.drop(['country'], axis=1, inplace=True)

# correlation_matrix(df)
# Not important features (after fit)
# df.drop(['country', 'admin_fee'], axis=1, inplace=True)

# ------------------------------------------ EDA dummy vars ------------------------------------------#

# You don't know the test set, thus number of columns in fitted model can be wrong!!!

# Check levels categorical features
# if not flag_unseen_data:
#     [(r'Feature: {}'.format(x), r'levels: {}'.format(df[x].value_counts().count())) for x in cat_features(df)]
#
# # Dummy encode
# dummy_features = [ftr for ftr in cat_features(df) if df[ftr].value_counts().count() < 60]
# df = pd.get_dummies(df, columns=dummy_features, drop_first=True)

# TODO Try extra categories to dummy (when model performance is not good enough)
# df = pd.get_dummies(df, columns=['inspector_name'], drop_first=True)
# df = pd.get_dummies(df, columns=['violation_code'], drop_first=True)


# ------------------------------------------ EDA impute NaN's ------------------------------------------#

# list(zip(df.columns, df.dtypes, df.isnull().any(), df.isnull().sum(), df.isnull().mean()))

# Replace infinte numbers
df.replace([np.inf, -np.inf], np.nan)

null_columns = list(df.columns[df.isnull().any()])
null_objects = cat_features(df[null_columns])
null_numerics = num_features(df[null_columns])

# Convert date features to date and forward fill NaN's
df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
[df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]

# Fill NaN's with '' for object columns
# na_objects = [column for column, isna in zip(df.columns, ((cat_features(df)) & (df.isnull().any()))) if isna]

[df.loc[:, x].fillna('', inplace=True) for x in null_objects] # TODO check otherwise go back to below
# [df.loc[:, x].fillna(df.loc[:, x].mode(), inplace=True) for x in null_objects]
# [df.loc[:, x].fillna(method='ffill', inplace=True) for x in na_objects]

# Fill NaN's with mean for numeric columns
# na_numerics = [column for column, isna in zip(df.columns, ((df.dtypes == 'float64') & (df.isnull().any()))) if isna]
[df.loc[:, x].fillna(df.loc[:, x].mean(), inplace=True) for x in null_numerics]

# Drop all above missed nan's
# df.dropna(axis=0, how='any', inplace=True)
# list(df.columns[df.isnull().any()])
# df.info()
# df['fine_amount'].describe()
assert df.isnull().sum().sum() == 0, 'Not all NaNs are removed'

# TODO impute bygroup or using Imputer (when not good enough)
# from sklearn.preprocessing import Imputer
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)


# ------------------------------------------ TODO outliers ------------------------------------------#
# http://scikit-learn.org/dev/auto_examples/ensemble/plot_isolation_forest.html
# #sphx-glr-auto-examples-ensemble-plot-isolation-forest-py


# ------------------------------------------ Feater selection ------------------------------------------#

# Mask or multi ti binary class
# not_class2 = df['compliance'] != 2

# Create X, y for final full model fit
# model_features = (set(num_features(df)) | set(date_features(df)))  # timestamp cannot be scaled
model_features = num_features(df)
# model_features = ['agency_name', 'inspector_name', 'state', 'zip_code', 'country',
#  'violation_description', 'fine_amount', 'admin_fee', 'discount_amount', 'judgment_amount']

model_features = ['agency_name', 'inspector_name', 'violation_description', 'discount_amount', 'judgment_amount']


def product(*args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def feature_selector(features):
    features = model_features
    selectors = list(product(range(2), repeat=len(features)))
    return list([[d for d, s in zip(features, s) if s] for s in selectors[1:]])

# ------------------------------------------ Split, transform dataset ------------------------------------------#

def split_transform(*args, phase='train'):
    X, y = args
    not_class2 = y != 2
    scaler = RobustScaler()  # StandardScaler()  # MinMaxScaler()
    if phase == 'train':
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Mulit class to binary
        X_train, X_test = X_train[np.array(y_train != 2)], X_test[np.array(y_test != 2)]
        y_train, y_test = y_train[np.array(y_train != 2)], y_test[np.array(y_test != 2)]
        return X_train, X_test, y_train, y_test
    if phase == 'model':
        X_model = scaler.fit_transform(X)
        X_model = X_model[np.array(not_class2)]  # syntax! [np.array[mask]]
        y_model = y[np.array(not_class2)]
        return X_model, y_model
    if phase == 'predict':
        X_unseen = scaler.fit_transform(X)
        return X_unseen

# ---------------------------------------------------------------------------------------------------------------------#
# ------------------------------------------ Data is preprocessed and ready! ------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

from sklearn.metrics import (auc, precision_recall_curve, roc_curve, roc_auc_score, precision_score)

# ------------------------------------------ Model selection ----------------------------------------------------------#

def model_performance(fitted_model, X_test, y_test, plot_cm=False):
    model_name = fitted_model.__class__.__name__
    predicted = fitted_model.predict(X_test)
    confusion = confusion_matrix(y_test, predicted)
    cm_dim = itemfreq(y_test)[:,0]
    df_cm = pd.DataFrame(confusion, index=cm_dim, columns=cm_dim)
    accuracy = accuracy_score(y_test, predicted)
    recall = recall_score(y_test, predicted, average='macro')
    precision = precision_score(y_test, predicted, average='macro')
    print('Test Accuracy: {:.2f}'.format(accuracy))
    print('Test Recall:   {:.2f}'.format(recall))
    print('Test Precision:   {:.2f}'.format(precision))
    try:
        auc = roc_auc_score(y_test, predicted, average='macro')
        print('Test AUC:   {:.2f}'.format(auc))
    except:
        print('No AUC for mulitclass')
    print('\nClassification Report:\n', classification_report(y_test, predicted))
    print('\nConfusion Matrix:\n', confusion)
    if plot_cm:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_cm, annot=True, cmap = 'RdBu_r', square=True)
        plt.title('Confusion Matrix: ' + model_name)
        plt.ylabel('True')
        plt.xlabel('Predicted')
    return model_name, accuracy, recall, precision

# ------------------------------------------ ROC and PR Curves --------------------------------------------------------#

def PR_ROC_curves(fitted_model, X_test, y_test, plot_curves=True):
    # from sklearn.metrics import (auc, precision_recall_curve, roc_curve, roc_auc_score, precision_score)
    y_scores = fitted_model.decision_function(X_test)
    fpr, recall_roc, thresholds = roc_curve(y_test, y_scores)
    precision, recall_pr, thresholds = precision_recall_curve(y_test, y_scores)
    roc_auc = auc(fpr, recall_roc)
    pr_auc = auc(recall_pr, precision)
    predicted = fitted_model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    roc_label = 'ROC AUC: {:.2f})'.format(roc_auc)
    pr_label = 'PR AUC: {:.2f})'.format(pr_auc)
    print(roc_label, pr_label)
    if plot_curves:
        plt.figure()
        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        title = 'ROC & PR curves: {}\nAccuracy: {:.2f}'.format(fitted_model.__class__.__name__, accuracy)
        plt.plot(fpr, recall_roc, label=roc_label)
        plt.plot(precision, recall_pr, label=pr_label)
        plt.plot([1, 1], [0, 0], 'k--')
        plt.title(title)
        plt.xlabel('FPR (ROC) / Precision (PR)', fontsize=12)
        plt.ylabel('Recall', fontsize=12)
        plt.legend(loc="lower left")
        plt.axes().set_aspect('equal')
        plt.show()
    return roc_auc, pr_auc

# ------------------------------------------ Feature Importance -------------------------------------------------------#

def feature_importance(fitted_model):
    importances = fitted_model.feature_importances_
    feature_importance = {ftr: imp for ftr, imp in zip(num_features(df), importances)}
    return sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# ------------------------------------------ Model scores -------------------------------------------------------------#

def model_selection(model, X_train, X_test, y_train, y_test):
    """"""
    # params = 'locals().values()'#locals().values()
    # print(model)
    params = model
    start_fit = time_lap()
    fit = model.fit(X_train, y_train)
    latency_fit = time_lap(start_fit)
    model_name = model.__class__.__name__
    print('=' * 60)
    print('\nModel: {}'.format(model_name))
    print('Parameters: {}'.format(params))
    print('Latency fit: {}'.format(latency_fit))
    start_pred = time_lap()
    model_name, accuracy, recall, precision = model_performance(fit, X_test, y_test)
    latency_pred = time_lap(start_pred)
    print('Latency predict: {}'.format(latency_pred))
    model_name = model.__class__.__name__
    feature_imp = None
    try:
        PR_ROC_curves(fit, X_test, y_test)
    except (ValueError, AttributeError):
        print('** PR & ROC curves are not available')
    try:
        feature_imp = feature_importance(fit)
    except (ValueError, AttributeError):
        print('** Feature_importance_ is not available')
    latency_ttl = time_lap(start_fit)
    print('Total time elapsed: {}'.format(latency_ttl))
    print('=' * 60)
    return [model_name, round(accuracy, 3), round(recall, 3), round(precision, 3), latency_fit, latency_pred,
            params, feature_imp]

def model_scores(selected_models, X_train, X_test, y_train, y_test):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_perf=[model_selection(model, X_train, X_test, y_train, y_test) for model in selected_models]
    return pd.DataFrame(model_perf, columns=['model_name', 'accuracy', 'recall', 'precision', 'fit_time', 'pred_time',
                                             'parameters', 'feature_importance'])

# ------------------------------------------ Model performance and selection ------------------------------------------#

# Import classifiers
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Fit models and check performance
selected_models = [
    # DummyClassifier(strategy='most_frequent').fit(X_train, y_train), # TODO error X_train
    # DummyClassifier(strategy='stratified').fit(X_train, y_train),
    GaussianNB(),
    LogisticRegression(random_state=0, penalty='l2'),
    DecisionTreeClassifier(max_depth=4, random_state=0),
    RandomForestClassifier(n_estimators=500, max_depth=5, n_jobs=-1, random_state=0),
    GradientBoostingClassifier(random_state=0),
    # KNeighborsClassifier(n_neighbors=5),
    # SVC(kernel = 'linear', random_state=0)
    ]
# model_scores1 = model_scores(selected_models)
selected_models2 = [
    # DecisionTreeClassifier(max_depth=2, random_state=0),
    # DecisionTreeClassifier(max_depth=4, random_state=0),
    # DecisionTreeClassifier(max_depth=6, random_state=0),
    # RandomForestClassifier(n_estimators=200, max_depth=2, n_jobs=-1, random_state=0),
    # RandomForestClassifier(n_estimators=200, max_depth=4, n_jobs=-1, random_state=0),
    # RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=-1, random_state=0),
    # GradientBoostingClassifier(learning_rate=0.05, random_state=0),
    # GradientBoostingClassifier(learning_rate=0.10, random_state=0),
    # GradientBoostingClassifier(learning_rate=0.25, random_state=0),
    # GradientBoostingClassifier(learning_rate=0.50, random_state=0),
    # KNeighborsClassifier(n_neighbors=2),
    KNeighborsClassifier(n_neighbors=4, n_jobs=-1),
    KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=6, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=8, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=9),
    ]

def train_model_grid(selected_models, model_features):
    feature_combinations = feature_selector(model_features)
    for model_ftrs in feature_combinations:
        print('#'*80)
        print('Featureset:', model_ftrs)
        X_train, X_test, y_train, y_test = split_transform(df.loc[:, model_ftrs].dropna(), df.loc[:, 'compliance'], phase='train')
        print(model_scores(selected_models, X_train, X_test, y_train, y_test))
        print('#' * 80)
    return None

# train_model_grid(selected_models2, model_features)

# ------------------------------------------ CV AUC scores ------------------------------------------#

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

def cv_scores(model, *args, k=5):
    X, y = args
    cv_scores = cross_val_score(model, X, y, cv=k, n_jobs=-1)
    print('Cross-validation Accuracies ({}-folds): {}\nMean Accuracy: {}'.format(k,
                            np.round_(cv_scores, 3), np.round_(np.mean(cv_scores), 3)))
    return None

def auc_scores(model, *args, k=3):
    X, y = args
    predictions = cross_val_predict(model, X, y, cv=k, n_jobs=-1)
    print('AUC: ', roc_auc_score(y, predictions))
    return None

#
model_features = ['violation_description', 'fine_amount', 'admin_fee', 'discount_amount', 'judgment_amount']
X_model, y_model = split_transform(df.loc[:, model_features].dropna(), df.loc[:, 'compliance'], phase='model')

kn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
auc_scores(kn, X_model, y_model)

nb = GaussianNB()
auc_scores(nb, X_model, y_model)

# lr = LogisticRegression(random_state=0) # crashes python
# auc_scores(lr, X_model, y_model)

dt = DecisionTreeClassifier(max_depth=4, random_state=0)
auc_scores(dt, X_model, y_model)

rf = RandomForestClassifier(n_estimators=200, max_depth=4, n_jobs=-1, random_state=0)
auc_scores(rf, X_model, y_model)

gb = GradientBoostingClassifier(learning_rate=0.05, max_depth=4, random_state=0)
auc_scores(gb, X_model, y_model)

sl = SVC(kernel = 'linear', random_state=0)
auc_scores(sl, X_model, y_model)

sr = SVC(kernel = 'rbf', random_state=0)
auc_scores(sr, X_model, y_model)

# sp = SVC(kernel = 'poly', random_state=0)  # system hangs
# cv_scores(sp, X_model, y_model)


# ----------------------------------------> learn final model <---------------------------------------- #

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(learning_rate=0.05, max_depth=2, random_state=0).fit(X_model, y_model)


# ----------------------------------------> predict final model <---------------------------------------- #
# TODO set to test phase

predicted = gb.predict(X_unseen)
pred_series = pd.DataFrame(predicted)
pred_series['ticket_id'] = df_unseen['ticket_id']
pred_series.set_index('ticket_id', inplace=True)






# ------------------------------------------ GridCV hyperparameter tuning ------------------------------------------#

from sklearn.model_selection import GridSearchCV

# default metric to optimize over grid parameters: accuracy
def grid_classifier(model, X_train, y_train, X_test, y_test, grid_values, scorings):
    best_score = 0
    best_param = {}
    for i, eval_metric in enumerate(scorings):
        clf = GridSearchCV(model, param_grid=grid_values, scoring=eval_metric, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_decision_scores = clf.decision_function(X_test)
        roc_score = roc_auc_score(y_test, y_decision_scores)
        score = '''Scoring metric: {}
                Test set AUC: {}
                Grid best parameter: {}
                Grid best score: {}'''.format(eval_metric, roc_score, clf.best_params_, clf.best_score_)
        print(score)
        if roc_score > best_score:
            best_score = roc_score
            best_param = clf.best_params_
    return best_param

# ------------------------------------------ Model hyperparameter tuning ------------------------------------------#

kn = KNeighborsClassifier().fit(X_train, y_train)
scorings_mc = ('roc_auc', 'f1_macro')                   # multi class
kn_grid = {'n_neighbors': [3, 5, 7, 10, 15]}
kn_params = grid_classifier(kn, X_train, y_train, X_test, y_test, kn_grid, scorings_mc)


# TODO error on multiclass
gb = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
scorings_bc = ('recall', 'f1', 'roc_auc', 'precision')  # binary class
scorings_mc = ('roc_auc', 'f1_macro')                   # multi class
gb_grid = {'learning_rate': [0.05, 0.1, 0.5, 1], 'max_depth': [2, 3, 4, 5, 6]}
gb_params = grid_classifier(gb, X_train, y_train, X_test, y_test, gb_grid, scorings_mc)



#------------------------------------------ Notes ------------------------------------------#

#
# Nice feature engineering snippets
# df_total = [df_train, df_test] # stacking together
# train['Name_length'] = train['Name'].apply(len)
# train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
# dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
#
#
# importances = gb.feature_importances_
# indices = np.argsort(importances)[::-1]

# g = sns.pairplot(train[['Survived', 'Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']],
# hue='Survived', palette = 'seismic', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
# g.set(xticklabels=[])

# Some useful parameters which will come in handy later on
# https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# ntrain = train.shape[0]
# ntest = test.shape[0]
# SEED = 0  # for reproducibility
# NFOLDS = 5  # set folds for out-of-fold prediction
# kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)
#

# import pandas.rpy.common as com
# load the R package ISLR
# infert = com.importr("ISLR")
#

# # Class to extend the Sklearn classifier
# class SklearnHelper(object):
#     def __init__(self, clf, seed=0, params=None):
#         params['random_state'] = seed
#         self.clf = clf(**params)
#
#     def train(self, x_train, y_train):
#         self.clf.fit(x_train, y_train)
#
#     def predict(self, x):
#         return self.clf.predict(x)
#
#     def fit(self, x, y):
#         return self.clf.fit(x, y)
#
#     def feature_importances(self, x, y):
#         print(self.clf.fit(x, y).feature_importances_)
#

#------------------------------------------ Featureset ------------------------------------------#

# You can set the type at read as dictionary
types = {'ticket_id',
         'agency_name',
         'inspector_name',
         'violator_name',
         'violation_street_number',
         'violation_street_name',
         'violation_zip_code',
         'mailing_address_str_number',
         'mailing_address_str_name',
         'city',
         'state',
         'zip_code',
         'non_us_str_code',
         'country',
         'ticket_issued_date',
         'hearing_date',
         'violation_code',
         'violation_description',
         'disposition',
         'fine_amount',
         'admin_fee',
         'state_fee',
         'late_fee',
         'discount_amount',
         'clean_up_cost',
         'judgment_amount',
         'payment_amount',
         'balance_due',
         'payment_date',
         'payment_status',
         'collection_status',
         'grafitti_status',
         'compliance_detail',
         'compliance'
 }


def PR_curve(fitted_model, X_test, y_test):
    from sklearn.metrics import auc
    y_scores = fitted_model.decision_function(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    auc = auc(recall, precision)
    predicted = fitted_model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)

    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='PR curve (area: {:.2f})\nAccuracy: {:.2f}'.format(auc, accuracy))
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle='none', c='r', mew=3)
    # plt.plot([0, 1], [0, 1], 'k--', c='r')
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.legend(loc="lower left")
    plt.axes().set_aspect('equal')
    plt.show()
    return auc


def ROC_curve(fitted_model, X_test, y_test):
    from sklearn.metrics import auc
    y_scores = fitted_model.decision_function(X_test)
    fpr, recall, thresholds = roc_curve(y_test, y_scores)
    auc = auc(fpr, recall)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = fpr[closest_zero]
    closest_zero_r = recall[closest_zero]

    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(fpr, recall, label='ROC curve (area = %0.2f)' % auc)
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle='none', c='r', mew=3)
    plt.plot([1, 1], [0, 0], 'k--', c='r')
    plt.xlabel('FPR - Specificity', fontsize=16)
    plt.ylabel('TPR - Recall', fontsize=16)
    plt.legend(loc="lower right")
    plt.axes().set_aspect('equal')
    plt.show()
    return auc


# --------------------------------------------->  <-------------------------------------------- #

print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [df.loc[:1000, ['fine_amount', 'judgment_amount', 'compliance']].dropna(),
            df.loc[:1000, ['admin_fee', 'judgment_amount', 'compliance']].dropna(),
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds.iloc[:, [0, 1]], ds.iloc[:, 2]#ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()