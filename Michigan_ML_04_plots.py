
# ------------------------------------------ Import libraries ------------------------------------------#
import numpy as np
import pandas as pd
import re
from time import time, gmtime, strftime
from scipy.stats import itemfreq
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------ Settings and CONSTANTS ---------------------------------------------------#

from matplotlib import rcParams
rcParams['figure.figsize'] = 12, 8

# ------------------------------------------ General Helper functions -------------------------------------------------#

def array2df(X_train, feature_names):
    """Convert np array to df, use with correlation matrix"""
    return pd.DataFrame(X_train, columns=feature_names)

def time_lap(start_time=None):
    """Stopwatch, No param = set, param (start) is time elapsed since start"""
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

def bin_cut(df, feature, bins, replace=0, cut=pd.qcut):
    return cut(df.loc[:, feature], bins, retbins=True)[0]

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
    df[cat_feature] = df[cat_feature].dropna(axis=0) #.fillna('missing')
    group = df.groupby([cat_feature, target])[target].count().unstack(level=0).T.fillna(0)
    return group

def top_cat(df_, feature, top=10):
    """Replace top 10 most frequent labels with 0-9 and rest with 10"""
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    labels = alphabet[:top]
    other = alphabet[top+1]
    top_violation_codes = df_.groupby(feature)[feature].count().sort_values(ascending=False).head(top).index.tolist()
    map_values = {k:l for k, l in (zip(top_violation_codes, labels))}  # [::-1]
    key_others = set(map_values.keys()) ^ (set(df_.loc[:, feature].values))
    map_others = {k:other for k in key_others}
    map_all = {**map_others, **map_values}
    df_.loc[:, feature] = df_.loc[:, feature].replace(map_all).astype('category')
    return df_

def plot_feature(df, feature, target='compliance'):
    cat_distribution(df, feature, target).plot(kind='barh', figsize=(12, 7))
    plt.xlabel('count')
    plt.show()

#------------------------------------------ Import datasets -----------------------------------------------------------#

# Import
set_types = {'compliance':object}
df_train = pd.read_csv('train.csv', encoding='latin-1', low_memory=False, dtype=set_types)
df_unseen = pd.read_csv('test.csv', encoding='latin-1', low_memory=False, dtype=set_types)
df_zip = pd.read_csv('zipcode.csv')
# df_zip.info()

# ---------------------------------------------> Merge datasets on zip_code <----------------------------------------- #

# zip_range = range(min(df_zip.loc[:,'zip']), max(df_zip.loc[:,'zip']))
# valid_zip_range = set(map(int, zip_range))
df_train.loc(1)['zip_code'] = numerize_code(df_train, 'zip_code', 99).astype('i')

# Add missing frequent zips
new_zips = [{'zip': 92714, 'city': 'Irvine', 'state': 'CA', 'latitude': 33.6881, 'longitude': -117.802,
           'timezone': -9, 'dst': 1},
            {'zip': 48033, 'city': 'Southfield', 'state': 'MI', 'latitude': 42.4723, 'longitude': -83.294,
           'timezone': -5, 'dst': 1},
            {'zip': 17202, 'city': 'Chambersburg', 'state': 'PA', 'latitude': 39.9072, 'longitude': -77.636,
             'timezone': -5, 'dst': 1},
            {'zip': 48193, 'city': 'Riverview', 'state': 'MI', 'latitude': 42.1782, 'longitude': -83.2461,
           'timezone': -5, 'dst': 1},
            {'zip': 63368, 'city': 'St. Charles', 'state': 'MO', 'latitude': 38.7513, 'longitude': -90.7296,
           'timezone': -5, 'dst': 1}]
df_zip = df_zip.append(new_zips, ignore_index=True)

# find outliers in zip_code
M = df_train.loc[:, 'zip_code'].isin(df_zip.loc[:,'zip'])  # symmetric_difference
df_train.loc[~M, 'zip_code'].value_counts().head(20)

# Most frequent zip_code for imputing
top_zips = df_train.loc[:, 'zip_code'].value_counts().head(25)
zip_samples = np.random.choice(top_zips.index, len(df_train.loc[~M, 'zip_code']))
df_train.loc[~M, 'zip_code'] = zip_samples

# Merge
df_merged = pd.merge(df_train, df_zip, how='left', left_on='zip_code', right_on='zip')
df_merged.loc(1)['zip'] = numerize_code(df_merged, 'zip', 99)
set(df_merged.loc[:, 'zip_code']).symmetric_difference(set(df_merged.loc[:, 'zip']))

# Generalize working phase
df = df_merged.copy(deep=True)
# df.info()

# ------------------------------------------ Feature Selection --------------------------------------------------------#

# Set y(target) to proper category type
df['compliance'] = df['compliance'].fillna(2).replace({'0.0': 0, '1.0': 1}).astype('category')
# Features
train_only_features = ['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status']
redundant_features = ['ticket_id', 'violation_street_name', 'violation_street_number', 'violation_zip_code',
                      'violation_description', 'mailing_address_str_name', 'mailing_address_str_number',
                      'non_us_str_code', 'grafitti_status','dst', 'timezone', 'compliance_detail', 'agency_name',
                      'inspector_name', 'violator_name', 'city_x', 'state_x', 'city_y', 'state_y', 'country', 'zip',
                      'zip_code', 'discount_amount', 'state_fee', 'clean_up_cost', 'discount_amount', 'admin_fee']
garbage_features = set(train_only_features) | set(redundant_features)

df.drop(garbage_features, axis=1, inplace=True)
df = df.dropna(axis=0)
# df.info()

# ---------------------------------------------> EDA & Feature Engineering <------------------------------------------ #

# Mask (minor) class 2 for plotting
trgt = 'compliance'
M = (df.loc[:, trgt] == 1)

# convert all dates features
df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
[df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]

# ---------------------------------------------> latlong <------------------------------------------------------------ #
# Geo, numerical
lat = df.loc[:, 'latitude']
long = df.loc[:, 'longitude']
sns.regplot(x=lat, y=long, data=df, dropna=True, fit_reg=False)
sns.regplot(x=lat, y=long, data=df[M], dropna=True, fit_reg=False, scatter_kws={'alpha':0.05})
plt.close()

# ---------------------------------------------> 'ticket_issued_date' <----------------------------------------------- #
# categorical - t_month, t_day_of_week
ftr = 'ticket_issued_date'
df['t_month'] = df.loc(1)[ftr].dt.month
df['t_day_of_week'] = df.loc(1)[ftr].dt.dayofweek
df.drop(ftr, axis=1, inplace=True)
ftr='t_month'
plot_feature(df, ftr)
ftr='t_day_of_week'
plot_feature(df, ftr)
plt.close()

# ---------------------------------------------> 'hearing_date' <----------------------------------------------------- #
# categorical - h_month, h_day_of_week
ftr = 'hearing_date'
df['h_month'] = df.loc(1)[ftr].dt.month
df['h_day_of_week'] = df.loc(1)[ftr].dt.dayofweek
df.drop(ftr, axis=1, inplace=True)
ftr='h_month'
plot_feature(df, ftr)
ftr='h_day_of_week'
plot_feature(df, ftr)
plt.close()

# ---------------------------------------------> 'violation_code' <--------------------------------------------------- #
# categorical
ftr = 'violation_code'
plot_feature(df, ftr)
# replace values of feature
top_cat(df, ftr, 10)
plot_feature(df, ftr)
cat_distribution(df, ftr, 'compliance')
plt.close()

# ---------------------------------------------> 'fine_amount' <------------------------------------------------------ #
# binary
ftr = 'fine_amount'
thres_min, thres_max = 20, 251
df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
ftr = 'thres'
plot_feature(df, ftr)
# replace values of feature
ftr = 'fine_amount'
df[ftr] = df['thres'].astype('int')
plt.close()

# ---------------------------------------------> 'judgment_amount' <-------------------------------------------------- #
# binary
ftr = 'judgment_amount'
thres_min, thres_max = 50, 300
df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
ftr = 'thres'
plot_feature(df, ftr)
# replace values of feature
ftr = 'judgment_amount'
df[ftr] = df['thres'].astype('int')
plt.close()

# ---------------------------------------------> 'late_fee' <--------------------------------------------------------- #
# binary
ftr = 'late_fee'
thres_min, thres_max = -0.1, 1
df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
ftr = 'thres'
plot_feature(df, ftr)
# replace values of feature
ftr = 'late_fee'
df[ftr] = df['thres'].astype('int')
plt.close()

# ---------------------------------------------> 'disposition' <------------------------------------------------------ #
# binary
ftr = 'disposition'
plot_feature(df, ftr)
# Replace values of feature
top_cat(df, ftr, top=1)
plot_feature(df, ftr)
plt.close()

# ---------------------------------------------> Last Feature Cleaning <---------------------------------------------- #

df.drop('thres', axis=1, inplace=True)

# close all plots
plt.close('all')
# df.info()

# ---------------------------------------------> Dummy vars <--------------------------------------------------------- #

df = pd.get_dummies(df, columns=['violation_code', 'disposition'], drop_first=True)

# ------------------------------------------ Final dataset ------------------------------------------------------------#

assert df.isnull().sum().sum() == 0, 'Not all NaNs are removed'
train_features = set(df.columns.tolist()) ^ set(['compliance'])
model_X = df.loc[:, train_features]
model_y = df.loc[:, 'compliance']

# ---------------------------------------------------------------------------------------------------------------------#
# ------------------------------------------ Train Features are ready -------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

# ---------------------------------------------> Import sklearn packages <-------------------------------------------- #

# Import preprocessing, selection and metrics
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

# Import classifiers
from sklearn.dummy import DummyClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
# from sklearn.svm import SVC


# ------------------------------------------ Split, transform, helper dataset -----------------------------------------#


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
    # features = model_features
    selectors = list(product(range(2), repeat=len(features)))
    return list([[d for d, s in zip(features, s) if s] for s in selectors[1:]])

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

# ---------------------------------------------> Model Perfomance and Selection <------------------------------------- #

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

def feature_importance(fitted_model):
    importances = fitted_model.feature_importances_
    feature_importance = {ftr: imp for ftr, imp in zip(num_features(df), importances)}
    return sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

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

def train_model_grid(selected_models, model_features):
    feature_combinations = feature_selector(model_features)
    for model_ftrs in feature_combinations:
        print('#'*80)
        print('Featureset:', model_ftrs)
        y = df.loc[:, 'compliance']
        X_train, X_test, y_train, y_test = split_transform(df.loc[:, model_ftrs].dropna(), y, phase='train')
        print(model_scores(selected_models, X_train, X_test, y_train, y_test))
        print('#' * 80)
    return None

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

# ---------------------------------------------> CV AUC <------------------------------------------------------------- #
#
X_model, y_model = split_transform(model_X, model_y, phase='model')

kn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
auc_scores(kn, X_model, y_model)

nb = GaussianNB()
auc_scores(nb, X_model, y_model)

dt = DecisionTreeClassifier(max_depth=4, random_state=0)
auc_scores(dt, X_model, y_model)

rf = RandomForestClassifier(n_estimators=200, max_depth=4, n_jobs=-1, random_state=0)
auc_scores(rf, X_model, y_model)

gb = GradientBoostingClassifier(learning_rate=0.05, max_depth=4, random_state=0)
auc_scores(gb, X_model, y_model)

# Too expensive / system hangs
# sl = SVC(kernel='linear') # sr = SVC(kernel='rbf') # sp = SVC(kernel='poly')

# ---------------------------------------------> Model Tuning <------------------------------------------------------- #

# Fit models and check performance
selected_models = [
    # DummyClassifier(strategy='most_frequent'),
    # DummyClassifier(strategy='stratified'),
    GaussianNB(),
    # DecisionTreeClassifier(max_depth=2, random_state=0),
    DecisionTreeClassifier(max_depth=4, random_state=0),
    # DecisionTreeClassifier(max_depth=6, random_state=0),
    # RandomForestClassifier(n_estimators=200, max_depth=2, n_jobs=-1, random_state=0),
    # RandomForestClassifier(n_estimators=200, max_depth=4, n_jobs=-1, random_state=0),
    # RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=-1, random_state=0),
    # GradientBoostingClassifier(learning_rate=0.05, random_state=0),
    # GradientBoostingClassifier(learning_rate=0.10, random_state=0),
    # GradientBoostingClassifier(learning_rate=0.25, random_state=0),
    # GradientBoostingClassifier(learning_rate=0.50, random_state=0),
    # KNeighborsClassifier(n_neighbors=2),
    # KNeighborsClassifier(n_neighbors=4, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=6, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=8, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=9),
    ]

# ---------------------------------------------> Model Scores All Feature Combinations <------------------------------ #

# train_model_grid(selected_models, train_features)

# ---------------------------------------------> Feature Selection and Fitting <-------------------------------------- #

X_model, y_model = split_transform(model_X, model_y, phase='model')

clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty='l1', loss='squared_hinge', dual=False))),
  ('classification', GaussianNB())
])
clf.fit(X_model, y_model)
auc_scores(clf, X_model, y_model)
# clf.get_params(deep=True)

# ---------------------------------------------> Final Model Fit <---------------------------------------------------- #

X_model, y_model = split_transform(model_X, model_y, phase='model')
nb = GaussianNB().fit(X_model, y_model)

# fit = GradientBoostingClassifier(learning_rate=0.05, max_depth=4, random_state=0).fit(X_model, y_model)

# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------> Test Final Model <-------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

test_features = ['fine_amount', 'late_fee', 'judgment_amount', 'ticket_issued_date', 'hearing_date', 'violation_code',
                 'zip_code', 'disposition']
df = df_unseen.loc[:, set(test_features)]

# find outliers in zip_code
df.loc[:, 'zip_code'] = numerize_code(df, 'zip_code', 99).astype('i')
M = df.loc[:, 'zip_code'].isin(df_zip.loc[:,'zip'])  # symmetric_difference
df.loc[~M, 'zip_code'].value_counts().head(20)

# Most frequent zip_code for imputing
top_zips = df.loc[:, 'zip_code'].value_counts().head(25)
zip_samples = np.random.choice(top_zips.index, len(df.loc[~M, 'zip_code']))
df.loc[~M, 'zip_code'] = zip_samples

# Merge on zip_code - add latlong
df_merged = pd.merge(df, df_zip, how='left', left_on='zip_code', right_on='zip')
df_merged.loc(1)['zip'] = numerize_code(df_merged, 'zip', 99)
df = df_merged

# ------------------------------------------ Feature engineering ------------------------------------------------------#

# convert all dates
df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
[df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]

# Geo, numerical
lat = df.loc[:, 'latitude']
long = df.loc[:, 'longitude']

# categorical - t_month, t_day_of_week
ftr = 'ticket_issued_date'

df['t_month'] = df.loc(1)[ftr].dt.month
df['t_day_of_week'] = df.loc(1)[ftr].dt.dayofweek
df.drop(ftr, axis=1, inplace=True)

# categorical - h_month, h_day_of_week
ftr = 'hearing_date'
# Impute nan
df.loc(1)[ftr] = df.loc(1)[ftr].fillna(method='pad')
df['h_month'] = df.loc(1)[ftr].dt.month
df['h_day_of_week'] = df.loc(1)[ftr].dt.dayofweek
df.drop(ftr, axis=1, inplace=True)

# categorical
ftr = 'violation_code'
# replace values of feature
top_cat(df, ftr, 10)

# binary
ftr = 'fine_amount'
thres_min, thres_max = 20, 251
df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
# replace values of feature
ftr = 'fine_amount'
df[ftr] = df['thres'].astype('int')

# binary
ftr = 'judgment_amount'
thres_min, thres_max = 50, 300
df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
# replace values of feature
ftr = 'judgment_amount'
df[ftr] = df['thres'].astype('int')

# binary
ftr = 'late_fee'
thres_min, thres_max = -0.1, 1
df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
# replace values of feature
ftr = 'late_fee'
df[ftr] = df['thres'].astype('int')

# categorical
ftr = 'disposition'
top_cat(df, ftr, top=1)

# ---------------------------------------------> Last Featureset Cleaning <------------------------------------------- #

df.drop(['city', 'dst', 'state', 'timezone', 'zip', 'zip_code', 'thres'], axis=1, inplace=True)

# ---------------------------------------------> Dummy vars <--------------------------------------------------------- #

df = pd.get_dummies(df, columns=['violation_code', 'disposition'], drop_first=True)

# ------------------------------------------ Final dataset ------------------------------------------------------------#

assert df.isnull().sum().sum() == 0, 'Not all NaNs are removed'
assert set(df.columns.tolist()) == set(train_features), 'Features train and test is not same'
test_features = set(df.columns.tolist())
model_X = df.loc[:, test_features]

# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------> Predict Final Model <----------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

X_unseen = split_transform(model_X, None, phase='predict')

predicted = nb.predict(X_unseen)
pred_series = pd.DataFrame(predicted)
pred_series['ticket_id'] = df_unseen['ticket_id']
pred_series.set_index('ticket_id', inplace=True)

pred_series

# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------> End <--------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

















def df_Xy(df_, features, target):
    """does not delete other features in df"""
    M1 = df_.loc[:, features].isnull()
    M2 = (df[target] != 2) & (~M1)
    scaler = MinMaxScaler()
    if isinstance(features, list):
        for feature in features:
            df_.loc[M2, feature] = scaler.fit_transform(df_.loc[M2, feature].values.reshape(-1, 1))
    else:
        df_.loc[M2, features] = scaler.fit_transform(df_.loc[M2, features].values.reshape(-1, 1))
    return df_

# M4 = df_1.loc[:, trgt].isin([0, 1])  # multivalue one-feature selection
# M5 = df_1.isin({trgt: [0, 1]})       # multivalue multi-feature selection, returns df mask
# df_1.isin(M4)

# ---------------------------------------------> Features with no information <-------------------------------------------- #
# ---------------------------------------------> 'agency_name' <-------------------------------------------- #

# little or no information
[plot_feature(df, f) for f in ['agency_name', 'inspector_name', 'violator_name', 'city_x', 'country', 'clean_up_cost']]

# ---------------------------------------------> 'state_x' <-------------------------------------------- #
# little/no information

ftr = 'state_x'
plot_feature(df, ftr)
# Replace values of feature
top_cat(df, ftr, top=5)
plot_feature(df, ftr)

# ---------------------------------------------> 'discount_amount' <-------------------------------------------- #
# no information

ftr = 'discount_amount'
thres_min = 20
thres_max = 300
df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
ftr = 'thres'
plot_feature(df, ftr)

# ---------------------------------------------> 'state_fee' <-------------------------------------------- #
# no information

ftr = 'state_fee'
thres_min = 0
thres_max = 9
df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
ftr = 'thres'
plot_feature(df, ftr)


# ---------------------------------------------> 'discount_amount' <-------------------------------------------- #
# no information

ftr = 'discount_amount'
thres_min = 1
thres_max = 25
df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
ftr = 'thres'
plot_feature(df, ftr)

# ---------------------------------------------> 'admin_fee' <-------------------------------------------- #
# no information

ftr = 'admin_fee'
thres_min = 0
thres_max = 19
df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
ftr = 'thres'
plot_feature(df, ftr)

# --------------------------------------------->  <-------------------------------------------- #

# ---------------------------------------------> judgment_amount <-------------------------------------------- #

ftr = 'judgment_amount'
ftr2 = 'fine_amount'
bins = 4
thres = 0.1

df_1 = df_Xy(df, ftr, trgt)
df_1['thres'] = df_1[ftr] < thres
df_1['bins'] = bin_cut(df_1, ftr, bins, cut=pd.cut)

M = df_1[ftr] < 1
M1 = (df_1[ftr] < thres)
M2 = (df_1[trgt] == 1)
M3 = M1 & M2

x = df_1[ftr]
x_zoom_1 = df_1.loc[M1, ftr]
x_zoom_2 = df_1.loc[M3, ftr]

# Distribution univariate - outliers
sns.distplot(x)
sns.kdeplot(x, bw=.01, label='bandwidth: 0.01')
plt.show()

# Zoom
sns.distplot(x_zoom_1, label='all')
sns.distplot(x_zoom_2, label='non-compliant')
sns.kdeplot(x_zoom_1, bw=.01, label='bandwidth: 0.01')
plt.show()

# Distribution by target
sns.violinplot(x=ftr, y=trgt, data=df_1[M], hue='thres', split=True)
plt.show()

# Correlogram
df_2 = df_1.loc(1)[ftr, trgt].dropna(axis=0)
sns.pairplot(df_2, kind='scatter', hue=trgt, plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.show()

# Bivariate numerical
sns.regplot(x=ftr, y=ftr2, data=df_1, dropna=True, fit_reg=False)
sns.regplot(x=ftr, y=ftr2, data=df_1[M2], dropna=True, fit_reg=False)
plt.show()

# --------------------------------------------->  <-------------------------------------------- #
# ---------------------------------------------> Univariate plots <-------------------------------------------- #



sns.distplot(x, bins=100)  # , rug=True is expensive
sns.distplot(x1, bins=None)
sns.distplot(x2)

sns.distplot(x)
sns.kdeplot(x, bw=.01, label='bandwidth: 0.01')

# ---------------------------------------------> TODO Pair plots <-------------------------------------------- #

# sns.pairplot(df_1.loc(1)[num_features(df_1)])  # axis=1
#
# g = sns.PairGrid(df_1)
# g.map_diag(sns.kdeplot)
# g.map_offdiag(sns.violinplot, cmap="Blues_d")

# ---------------------------------------------> Categorial plots <-------------------------------------------- #

sns.violinplot(x='judgment_amount', y='compliance', data=df_1[M])
sns.violinplot(x='judgment_amount', y='compliance', data=df_1[M], scale='count')
sns.violinplot(x='judgment_amount', y='compliance', data=df_1[M], hue='special', scale_hue=False)
sns.violinplot(x='judgment_amount', y='compliance', data=df_1[M], hue='special', split=True)
sns.boxplot(x='judgment_amount', y='compliance', hue='special', data=df_1)
# judgment_amount below 0.4 similar distributions for each label, above only 0

sns.barplot(x='bins', y='judgment_amount', hue='compliance', data=df_1)
sns.countplot(x='bins', hue='compliance', data=df_1, palette='Blues_d')

sns.factorplot(x='bins', y='judgment_amount', hue='compliance', data=df_1, palette='Blues_d', kind='violin')
sns.factorplot(x='compliance', y='judgment_amount', hue='compliance',
               col='bins', data=df_1, kind='box', size=4, aspect=.5)

# ---------------------------------------------> Regression plots <-------------------------------------------- #

df_2 = df_Xy(df, ['judgment_amount', 'fine_amount'], 'compliance')
df_2['special'] = df_1['judgment_amount'] < 0.1
M = df_2['judgment_amount'] < 1
df_2['bins'] = bin_cut(df_1, 'judgment_amount', 4, cut=pd.cut)

M1 = (df_2['judgment_amount'] < 0.06)
M2 = (df_2['fine_amount'] < 1000) & M1
x = df_2['judgment_amount']
x1 = df_2.loc[M1, 'judgment_amount']
x2 = df_2.loc[M2, 'judgment_amount']

sns.regplot(x="judgment_amount", y='fine_amount', data=df_2, dropna=True, fit_reg=False)
sns.regplot(x="judgment_amount", y='fine_amount', data=df_2)
sns.lmplot(x="judgment_amount", y='fine_amount', data=df_1)

# --------------------------------------------->  <-------------------------------------------- #





N = 10000
M = df.loc[:, 'compliance'] == 1



# Univariate - distribution
X_all = scaler.fit_transform(df.loc[:, 'judgment_amount'].values.reshape(-1, 1))
X_1 = scaler.transform(df.loc[M, 'judgment_amount'].values.reshape(-1, 1))
sns.distplot(X_all, bins=30, label='compliant')
sns.distplot(X_1, bins=30, label='compliant')

#
df.loc[M, 'judgment_amount'].plot(kind='hist')
df.loc[M, 'judgment_amount'].plot(kind='hist')

df.loc[:, 'judgment_amount'].plot(kind='kde')
df.loc[M, 'judgment_amount'].plot(kind='kde')

df.loc[:, 'judgment_amount'].plot(kind='box')
df.loc[M, 'judgment_amount'].plot(kind='box')


df.loc[:N, ['compliance', 'judgment_amount']].groupby('compliance').plot(kind='hist')
df.loc[:N, ['compliance', 'judgment_amount']].sort_values('compliance').set_index('compliance').plot(kind='hist')

df.loc[:N, 'judgment_amount'].plot.hist()  # same
df.loc[:N, 'judgment_amount'].plot(kind='density')


# Multivariate
df.loc[:N, ['compliance', 'judgment_amount']].sort_values('compliance').set_index('compliance').groupby('compliance').plot(kind='bar')  # index are the bins on x-as are columns
df.loc[:N, ['compliance', 'judgment_amount']].groupby('compliance').plot(kind='bar')
df.loc[:N, 'judgment_amount'].plot(kind='barh')  # expensive, crowded
df.loc[:N, ['compliance', 'judgment_amount']].plot.bar(stacked=True)

# Multivariate
df.loc[:N, 'judgment_amount'].plot(kind='hexbin')
df.loc[:N, 'judgment_amount'].plot(kind='scatter')
df.loc[:N, 'judgment_amount'].plot(kind='line')
df.loc[:N, 'judgment_amount'].plot()  # is line plot (numeric only)

plot_types = ['area', 'barh', 'density', 'hist', 'line', 'scatter', 'bar', 'box', 'hexbin', 'kde']  # pie
for type in plot_types:
    df.loc[:N, 'judgment_amount'].plot(kind=type)

#
# # ------------------------------------------ Visual inspection ------------------------------------------#
# # model_features = ['violation_description', 'fine_amount', 'admin_fee', 'discount_amount', 'judgment_amount', 'compliance']
# # X_model, y_model = split_transform(df.loc[:, model_features], df.loc[:, 'compliance'], phase='model')
#
# # Visualize raw features
# df['state'] = factorized(df['state'])
#
# # Show information of feature:
# # - distribution; outliers,
# # - relationship with target; are clusters indicative for feature importance
#
# # 2 features, with colored target
# # - what type of model/kernels are most effective; linear, polynomial, radial
#
# # # ---------------------------------------------> plot dataset <-------------------------------------------- #
#
#
# # --------------------------------------------->  <-------------------------------------------- #
#
# # fine_amount vs compliance
# # fine amount, violation_desc vs compliance
#
# df['fine_amount'].mean()
# high = df[df['fine_amount'] > 400]
# low = df[df['fine_amount'] < 400]
# high.mean()
#
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
# ax = sns.distplot(df[df['compliance'] == 1].violation_description.dropna(), bins=30, label='compliant', ax=axes[0])
# ax = sns.distplot(df[df['compliance'] == 0].violation_description.dropna(), bins=30, label='not compliant', ax=axes[0])
# ax.legend()
# ax.set_title('Comliance vs Violation description')
#
# ax = sns.distplot(df[df['compliance'] == 1].judgment_amount.dropna(), bins=30, label='compliant', ax=axes[1])
# ax = sns.distplot(df[df['compliance'] == 0].judgment_amount.dropna(), bins=30, label='not compliant', ax=axes[1])
# ax.legend()
# ax.set_title('Comliance vs Judgment amount')
#
# ax = sns.distplot(df[df['compliance'] == 1].fine_amount.dropna(), bins=30, label='compliant', ax=axes[2])
# ax = sns.distplot(df[df['compliance'] == 0].fine_amount.dropna(), bins=30, label='not compliant', ax=axes[2])
# ax.set_title('Comliance vs Fine amount')
# ax.legend()
# _ = ax.set_title(' ')
#
#
# #
# # def bar_plot(x, y):
# #     sns.barplot(x=x, y=y, data=df.fillna(method='pad'), linewidth=2)
# #     plt.title(str(x) + ' vs. Compliance', fontsize=16)
# #     plt.xlabel(str(x), fontsize=8)
# #     plt.ylabel('Compliance', fontsize=8)
# #
# # # numerical_features = df.select_dtypes(include=np.number).columns.tolist()
# # # bar_plot(non_generalizing_features[-1], y)
#
# ['violation_zip_code', 'non_us_str_code', 'payment_date', 'collection_status', 'grafitti_status']
#
# df.set_index(['compliance', 'judgment_amount'])
# df.groupby(level=('compliance', 'judgement_amount'))
#
#
# def scat_plot(df_):
#     na_features = ['violation_zip_code', 'non_us_str_code', 'payment_date', 'collection_status', 'grafitti_status']
#     df_ = df.loc[:, (set(df.columns) ^ set(na_features))]
#     df_ = df_[df_.iloc[:, 2] != 2].dropna(axis=0)
#     # df_ = df_.loc[:, num_features(df_)]
#     scatter_matrix(df_, alpha=0.2, figsize=(12, 8), diagonal='kde')
#
#     try:
#         check_df = df_.iloc[1, :]
#         X1, X2, y = df_.iloc[:, 0], df_.iloc[:, 1], df_.iloc[:, 2]
#         if df_.iloc[:, 0].dtypes != 'int' or 'float':
#             X1 = factorized(df_.iloc[:, 0])
#         if df_.iloc[:, 1].dtypes != 'int' or 'float':
#             X2 = factorized(df_.iloc[:, 1])
#         # plt.figure()
#         # s, alpha = 50, 0.6
#         # plt.scatter(X1, X2, alpha=alpha, c=y, label=df_.columns[2], s=s)
#         # plt.xlabel(df_.columns[0], fontsize=16)
#         # plt.ylabel(df_.columns[1], fontsize=16)
#         # plt.legend()
#         # plt.show()
#     except IndexError:
#         print('Empty df')
#
# # scat_plot(df.loc[:, ['compliance', 'violation_zip_code', 'compliance']])
#
# # scat_plot(df.loc[:, ['judgment_amount', 'violation_zip_code', 'compliance']])
# scat_plot(df.loc[:, ['judgment_amount', 'non_us_str_code', 'compliance']])
# scat_plot(df.loc[:, ['judgment_amount', 'payment_date', 'compliance']])
# scat_plot(df.loc[:, ['judgment_amount', 'collection_status', 'compliance']])
# scat_plot(df.loc[:, ['judgment_amount', 'grafitti_status', 'compliance']])
#
# from pandas.plotting import scatter_matrix
# scatter_matrix(df.dropna(axis=0), alpha=0.2, figsize=(6, 6), diagonal='kde')
#
#
# df.set_index('judgment_amount').plot(subplots=True, figsize=(6, 6))
#
#
# def scat_pred(fitted_model, X_train, y_train, X_test, y_test, X1, X2):
#     X_pred = np.random.random_sample((60,)).reshape(-1, 6)
#     y_pred = fitted_model.predict(X_pred)
#
#     plt.figure()
#     alpha = 0.6
#     s = 50
#     plt.scatter(X_train[:, X1], X_train[:, X2], marker='o', alpha=alpha, c=y_train, s=s)
#     plt.scatter(X_test[:, X1], X_test[:, X2], marker='^', alpha=alpha, c=y_test, s=s)
#     plt.scatter(X_pred[:, X1], X_pred[:, X2], marker='+', alpha=alpha, c=y_pred, s=s)
#     plt.show()
#
# scat_pred(gb, X_train, y_train, X_test, y_test, 4, 4)

# ------------------------------------------ Visual inspection 2 ------------------------------------------#
#
# def feature_distribution(X):
#     plt.figure()
#     plt.violinplot(X)
#     plt.show()
#
# feature_distribution(X_train)
# feature_distribution(X_test)
#
# sns.distplot(X_train[:, 1])
#
# df_ = array2df(X_test, num_features(df))
# sns.jointplot(x=df_.columns[0], y=df_.columns[1], data=df_)
#
# sns.set(style="white")
# g = sns.PairGrid(df_, diag_sharey=False)
# # g.map_lower(sns.kdeplot, cmap="Blues_d")
# g.map_upper(plt.scatter)
# g.map_diag(sns.kdeplot, lw=3)
#
#
# sns.set(style="darkgrid")
# g = sns.FacetGrid(df_, col=y_test,  row=y_test)
# g = g.map(plt.hist, df_.columns[1])


#------------------------------------------ MultiIndex slicing ------------------------------------------#
# http://pandas.pydata.org/pandas-docs/stable/advanced.html
# midx = pd.MultiIndex(levels=[['zero', 'one'], ['x','y']], labels=[[1,1,0,0],[1,0,1,0]])
# dfmi = pd.DataFrame(np.random.randn(4,2), index=midx)
# dfm = pd.DataFrame({'jim': [0, 0, 1, 1],
#                     'joe': ['x', 'x', 'z', 'y'],
#                     'jolie': np.random.rand(4)})
# dfm = dfm.set_index(['jim', 'joe'])
# dfmi.loc(axis=0)[:, :, ['C1', 'C3']]
# df.xs('one', level='second')  # cross section
# df.xs('one', level='second', axis=1)
# df.xs('one', level='second', axis=1, drop_level=False)