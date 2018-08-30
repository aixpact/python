#
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import re
# from sklearn.preprocessing import RobustScaler
#
#
# #------------------------------------------ General Helper functions ------------------------------------------#
#
# def num_features(df):
#     return df.select_dtypes(include=[np.number]).columns.tolist()
#
# def cat_features(df):
#     return df.select_dtypes(include=['object', 'category']).columns
#
# def date_features(df):
#     return df.columns.str.extractall(r'(.*date.*)')[0].values.tolist()
#
# def clip_outliers(values, p=99):
#     """clip at 1.5 IQR"""
#     min = np.percentile(values, 100-p)
#     max = np.percentile(values, p)
#     return np.clip(values, min, max)
#
# def numerize_code(df, feature, replace=0):
#     df[feature] = pd.to_numeric(df[feature], errors='coerce')
#     nums = df[feature].fillna(replace).astype('int64')
#     return clip_outliers(nums)
#
# def alpharize_code(df, feature, bins, replace=0, cut=pd.qcut, upper=True):
#     zipcode = numerize_code(df, feature, replace)
#     labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:bins]
#     if not upper:
#         labels = list(map(lambda x:x.lower(), labels))
#     return cut(zipcode, bins, labels=labels)
#
# def factorized(df):
#     if isinstance(df, pd.Series):
#         return df.factorize()[0]
#     return df.loc[:, cat_features(df)].apply(lambda x: x.factorize()[0])
#
# def correlation(df, features):
#     return df.loc[:, features].apply(lambda x: x.factorize()[0]).corr()
#
# def show_hist(df, features):
#     c, r, h = features
#     g = sns.FacetGrid(df, col=c,  row=r)
#     g = g.map(plt.hist, h)
#
# def cat_distribution(df, cat_feature, target):
#     df[cat_feature] = df[cat_feature].fillna('missing')
#     group = df.groupby([cat_feature, target])[target].count().unstack(level=0).T.fillna(0)
#     return group
#
# #------------------------------------------ Import dataset ------------------------------------------#
#
# # Import
# set_types = {'compliance':object}
# df_train = pd.read_csv('train.csv', encoding='latin-1', low_memory=False, dtype=set_types)
# df_unseen = pd.read_csv('test.csv', encoding='latin-1', low_memory=False, dtype=set_types)
#
# df = df_train.copy(deep=True)
#
# # Set y(target) to proper category type
# df['compliance'] = df['compliance'].fillna(2).replace({'0.0': 0, '1.0': 1}).astype('category')
#
# #------------------------------------------ Garbage Feature Collection ------------------------------------------#
#
# garbage_features = ['city', 'clean_up_cost', 'disposition', 'grafitti_status', 'hearing_date', 'late_fee',
#  'mailing_address_str_name', 'mailing_address_str_number', 'non_us_str_code', 'state_fee', 'ticket_id',
#  'violation_code', 'violation_street_name', 'violation_street_number', 'violation_zip_code', 'violator_name',
#  'payment_amount', 'balance_due']
#
# # EDA Clean and split featureset
# df.drop(garbage_features, axis=1, inplace=True)
#
# #------------------------------------------ Feature engineering ------------------------------------------#
#
# # Factorize in levels
# df['zip_code'] = factorized(df['zip_code'])
# df['agency_name'] = factorized(df['agency_name'])
# df['inspector_name'] = factorized(df['inspector_name'])
# df['violation_description'] = factorized(df['violation_description'])
#
# # Detroit yes/no
# df['state'] = factorized(df['state'])
# df.loc[df['state'] != 0, 'state'] = 1
#
# # USA yes/no
# df.loc[df['country'] == 'USA', 'country'] = 1
# df.loc[df['country'] != 1, 'country'] = 0
#
# #------------------------------------------ EDA impute NaN's ------------------------------------------#
#
# # Replace infinte numbers
# df.replace([np.inf, -np.inf], np.nan)
#
# null_columns = list(df.columns[df.isnull().any()])
# null_objects = cat_features(df[null_columns])
# null_numerics = num_features(df[null_columns])
#
# # Convert date features to date and forward fill NaN's
# df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
# [df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]
#
# # Fill NaN's with mean for categorical/numeric columns
# [df.loc[:, x].fillna('', inplace=True) for x in null_objects]
# [df.loc[:, x].fillna(df.loc[:, x].mean(), inplace=True) for x in null_numerics]
#
# assert df.isnull().sum().sum() == 0, 'Not all NaNs are removed'
#
# #------------------------------------------ Split dataset ------------------------------------------#
#
# # Split numerical features
# not_class2 = df['compliance'] != 2
# X_model1 = df.loc[:, num_features(df)]
# y_model = df.loc[not_class2, 'compliance']
#
# # ------------------------------------------ Scaling, transforming features ------------------------------------------#
#
# # Scaling incl. class 2?
# scaler = RobustScaler()  # StandardScaler()  # MinMaxScaler()
# X_model2 = scaler.fit_transform(X_model1)
# X_model = X_model2[not_class2]
#
# # ----------------------------------------> learn final model <---------------------------------------- #
#
# from sklearn.ensemble import GradientBoostingClassifier
# # gb = GradientBoostingClassifier(learning_rate=0.05, max_depth=2, random_state=0).fit(X_model, y_model)
#
# def cv_scores(model, X_train, y_train, k=10):
#     cv_scores = cross_val_score(model, X_train, y_train, cv=k, n_jobs=-1)
#     print('Cross-validation Accuracies ({}-folds): {}\nMean Accuracy: {}'.format(k,
#                             np.round_(cv_scores, 3), np.round_(np.mean(cv_scores), 3)))
#     return None
#
# from sklearn.model_selection import cross_val_score
# gb = GradientBoostingClassifier(learning_rate=0.05, max_depth=2, random_state=0)
# cv_scores(gb, X_model, y_model, 3)
#
# from sklearn.neighbors import KNeighborsClassifier
# kn = KNeighborsClassifier(n_neighbors=2)
# cv_scores(kn, X_model, y_model, 3)
#
#
#
# # ----------------------------------------> predict final model <---------------------------------------- #
# # ----------------------------------------> predict final model <---------------------------------------- #
#
# df = df_unseen.copy(deep=True)
#
# garbage_features = ['city', 'clean_up_cost', 'disposition', 'grafitti_status', 'hearing_date', 'late_fee',
#  'mailing_address_str_name', 'mailing_address_str_number', 'non_us_str_code', 'state_fee', 'ticket_id',
#  'violation_code', 'violation_street_name', 'violation_street_number', 'violation_zip_code', 'violator_name']
#
# # EDA Clean and split featureset
# df.drop(garbage_features, axis=1, inplace=True)
#
# #------------------------------------------ Feature engineering ------------------------------------------#
#
# # Factorize in levels
# df['zip_code'] = factorized(df['zip_code'])
# df['agency_name'] = factorized(df['agency_name'])
# df['inspector_name'] = factorized(df['inspector_name'])
# df['violation_description'] = factorized(df['violation_description'])
#
# # Detroit yes/no
# df['state'] = factorized(df['state'])
# df.loc[df['state'] != 0, 'state'] = 1
#
# # USA yes/no
# df.loc[df['country'] == 'USA', 'country'] = 1
# df.loc[df['country'] != 1, 'country'] = 0
#
# #------------------------------------------ EDA impute NaN's ------------------------------------------#
#
# # Replace infinte numbers
# df.replace([np.inf, -np.inf], np.nan)
#
# null_columns = list(df.columns[df.isnull().any()])
# null_objects = cat_features(df[null_columns])
# null_numerics = num_features(df[null_columns])
#
# # Convert date features to date and forward fill NaN's
# df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
# [df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]
#
# # Fill NaN's with mean for categorical/numeric columns
# [df.loc[:, x].fillna('', inplace=True) for x in null_objects]
# [df.loc[:, x].fillna(df.loc[:, x].mean(), inplace=True) for x in null_numerics]
#
# assert df.isnull().sum().sum() == 0, 'Not all NaNs are removed'
#
# #------------------------------------------ Scaling, transforming features ------------------------------------------#
#
# df_model = df.loc[:, num_features(df)]
# scaler = RobustScaler()
# X_unseen = scaler.fit_transform(df_model)
#
# predicted = gb.predict(X_unseen)
# pred_series = pd.DataFrame(predicted)
# pred_series['ticket_id'] = df_unseen['ticket_id']
# pred_series.set_index('ticket_id', inplace=True)
#
#
# # ============
#
# import pandas as pd
# import numpy as np
#
#
# def blight_model():
#     # Your code here
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import re
#     from sklearn.preprocessing import RobustScaler
#
#     # ------------------------------------------ General Helper functions ------------------------------------------#
#
#     def num_features(df):
#         return df.select_dtypes(include=[np.number]).columns.tolist()
#
#     def cat_features(df):
#         return df.select_dtypes(include=['object', 'category']).columns
#
#     def date_features(df):
#         return df.columns.str.extractall(r'(.*date.*)')[0].values.tolist()
#
#     def clip_outliers(values, p=99):
#         """clip at 1.5 IQR"""
#         min = np.percentile(values, 100 - p)
#         max = np.percentile(values, p)
#         return np.clip(values, min, max)
#
#     def numerize_code(df, feature, replace=0):
#         df[feature] = pd.to_numeric(df[feature], errors='coerce')
#         nums = df[feature].fillna(replace).astype('int64')
#         return clip_outliers(nums)
#
#     def alpharize_code(df, feature, bins, replace=0, cut=pd.qcut, upper=True):
#         zipcode = numerize_code(df, feature, replace)
#         labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:bins]
#         if not upper:
#             labels = list(map(lambda x: x.lower(), labels))
#         return cut(zipcode, bins, labels=labels)
#
#     def factorized(df):
#         if isinstance(df, pd.Series):
#             return df.factorize()[0]
#         return df.loc[:, cat_features(df)].apply(lambda x: x.factorize()[0])
#
#     def cat_distribution(df, cat_feature, target):
#         df[cat_feature] = df[cat_feature].fillna('missing')
#         group = df.groupby([cat_feature, target])[target].count().unstack(level=0).T.fillna(0)
#         return group
#
#     # ------------------------------------------ Import dataset ------------------------------------------#
#
#     # Import
#     set_types = {'compliance': object}
#     df_train = pd.read_csv('train.csv', encoding='latin-1', low_memory=False, dtype=set_types)
#     df_unseen = pd.read_csv('test.csv', encoding='latin-1', low_memory=False, dtype=set_types)
#
#     df = df_train.copy(deep=True)
#
#     # Set y(target) to proper category type
#     df['compliance'] = df['compliance'].fillna(2).replace({'0.0': 0, '1.0': 1}).astype('category')
#
#     # ------------------------------------------ Garbage Feature Collection ------------------------------------------#
#
#     garbage_features = ['city', 'clean_up_cost', 'disposition', 'grafitti_status', 'hearing_date', 'late_fee',
#                         'mailing_address_str_name', 'mailing_address_str_number', 'non_us_str_code', 'state_fee',
#                         'ticket_id',
#                         'violation_code', 'violation_street_name', 'violation_street_number', 'violation_zip_code',
#                         'violator_name',
#                         'payment_amount', 'balance_due']
#
#     # EDA Clean and split featureset
#     df.drop(garbage_features, axis=1, inplace=True)
#
#     # ------------------------------------------ Feature engineering ------------------------------------------#
#
#     # Factorize in levels
#     df['zip_code'] = factorized(df['zip_code'])
#     df['agency_name'] = factorized(df['agency_name'])
#     df['inspector_name'] = factorized(df['inspector_name'])
#     df['violation_description'] = factorized(df['violation_description'])
#
#     # Detroit yes/no
#     df['state'] = factorized(df['state'])
#     df.loc[df['state'] != 0, 'state'] = 1
#
#     # USA yes/no
#     df.loc[df['country'] == 'USA', 'country'] = 1
#     df.loc[df['country'] != 1, 'country'] = 0
#
#     # ------------------------------------------ EDA impute NaN's ------------------------------------------#
#
#     null_columns = list(df.columns[df.isnull().any()])
#     null_objects = cat_features(df[null_columns])
#     null_numerics = num_features(df[null_columns])
#
#     # Convert date features to date and forward fill NaN's
#     df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
#     [df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]
#
#     # Fill NaN's with mean for categorical/numeric columns
#     [df.loc[:, x].fillna('', inplace=True) for x in null_objects]
#     [df.loc[:, x].fillna(df.loc[:, x].mean(), inplace=True) for x in null_numerics]
#
#     assert df.isnull().sum().sum() == 0, 'Not all NaNs are removed'
#
#     # ------------------------------------------ Split dataset ------------------------------------------#
#
#     # Split numerical features
#     not_class2 = df['compliance'] != 2
#     X_model = df.loc[:, num_features(df)]
#     y_model = df.loc[not_class2, 'compliance']
#
#     # ------------------------------------------ Scaling, transforming features ------------------------------------------#
#
#     # Scaling incl. class 2?
#     scaler = RobustScaler()  # StandardScaler()  # MinMaxScaler()
#     X_model = scaler.fit_transform(X_model)
#     X_model = X_model[np.array(not_class2)]
#
#     # ----------------------------------------> learn final model <---------------------------------------- #
#
#     from sklearn.ensemble import GradientBoostingClassifier
#     gb = GradientBoostingClassifier(learning_rate=0.05, max_depth=2, random_state=0).fit(X_model, y_model)
#
#     # ----------------------------------------> predict final model <---------------------------------------- #
#     # ----------------------------------------> predict final model <---------------------------------------- #
#
#     df = df_unseen.copy(deep=True)
#
#     garbage_features = ['city', 'clean_up_cost', 'disposition', 'grafitti_status', 'hearing_date', 'late_fee',
#                         'mailing_address_str_name', 'mailing_address_str_number', 'non_us_str_code', 'state_fee',
#                         'ticket_id',
#                         'violation_code', 'violation_street_name', 'violation_street_number', 'violation_zip_code',
#                         'violator_name']
#
#     # EDA Clean and split featureset
#     df.drop(garbage_features, axis=1, inplace=True)
#
#     # ------------------------------------------ Feature engineering ------------------------------------------#
#
#     # Factorize in levels
#     df['zip_code'] = factorized(df['zip_code'])
#     df['agency_name'] = factorized(df['agency_name'])
#     df['inspector_name'] = factorized(df['inspector_name'])
#     df['violation_description'] = factorized(df['violation_description'])
#
#     # Detroit yes/no
#     df['state'] = factorized(df['state'])
#     df.loc[df['state'] != 0, 'state'] = 1
#
#     # USA yes/no
#     df.loc[df['country'] == 'USA', 'country'] = 1
#     df.loc[df['country'] != 1, 'country'] = 0
#
#     # ------------------------------------------ EDA impute NaN's ------------------------------------------#
#
#     null_columns = list(df.columns[df.isnull().any()])
#     null_objects = cat_features(df[null_columns])
#     null_numerics = num_features(df[null_columns])
#
#     # Convert date features to date and forward fill NaN's
#     df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
#     [df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]
#
#     # Fill NaN's with mean for categorical/numeric columns
#     [df.loc[:, x].fillna('', inplace=True) for x in null_objects]
#     [df.loc[:, x].fillna(df.loc[:, x].mean(), inplace=True) for x in null_numerics]
#
#     assert df.isnull().sum().sum() == 0, 'Not all NaNs are removed'
#
#     # ------------------------------------------ Scaling, transforming features ------------------------------------------#
#
#     df_model = df.loc[:, num_features(df)]
#     scaler = RobustScaler()
#     X_unseen = scaler.fit_transform(df_model)
#
#     predicted = gb.predict(X_unseen)
#     pred_series = pd.DataFrame(predicted)
#     pred_series['ticket_id'] = df_unseen['ticket_id']
#     pred_series.set_index('ticket_id', inplace=True)
#
#     return pred_series
#
# ##################################
# # --------------------------------------------->  <-------------------------------------------- #
# import pandas as pd
# import numpy as np
#
#
# def blight_model():
#     # Your code here
#     # import re
#     from sklearn.preprocessing import RobustScaler
#
#     # ------------------------------------------ General Helper functions ------------------------------------------#
#
#     def num_features(df):
#         return df.select_dtypes(include=[np.number]).columns.tolist()
#
#     def cat_features(df):
#         return df.select_dtypes(include=['object', 'category']).columns
#
#     def date_features(df):
#         return df.columns.str.extractall(r'(.*date.*)')[0].values.tolist()
#
#     def clip_outliers(values, p=99):
#         """clip at 1.5 IQR"""
#         min = np.percentile(values, 100 - p)
#         max = np.percentile(values, p)
#         return np.clip(values, min, max)
#
#     def numerize_code(df, feature, replace=0):
#         df[feature] = pd.to_numeric(df[feature], errors='coerce')
#         nums = df[feature].fillna(replace).astype('int64')
#         return clip_outliers(nums)
#
#     def alpharize_code(df, feature, bins, replace=0, cut=pd.qcut, upper=True):
#         zipcode = numerize_code(df, feature, replace)
#         labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:bins]
#         if not upper:
#             labels = list(map(lambda x: x.lower(), labels))
#         return cut(zipcode, bins, labels=labels)
#
#     def factorized(df):
#         if isinstance(df, pd.Series):
#             return df.factorize()[0]
#         return df.loc[:, cat_features(df)].apply(lambda x: x.factorize()[0])
#
#     def cat_distribution(df, cat_feature, target):
#         df[cat_feature] = df[cat_feature].fillna('missing')
#         group = df.groupby([cat_feature, target])[target].count().unstack(level=0).T.fillna(0)
#         return group
#
#     # ------------------------------------------ Import dataset ------------------------------------------#
#
#     # Import
#     set_types = {'compliance': object}
#     df_train = pd.read_csv('train.csv', encoding='latin-1', low_memory=False, dtype=set_types)
#     df_unseen = pd.read_csv('test.csv', encoding='latin-1', low_memory=False, dtype=set_types)
#
#     df = df_train.copy(deep=True)
#
#     # Set y(target) to proper category type
#     df['compliance'] = df['compliance'].fillna(2).replace({'0.0': 0, '1.0': 1}).astype('category')
#
#     # ------------------------------------------ Garbage Feature Collection ------------------------------------------#
#
#     garbage_features = ['city', 'clean_up_cost', 'disposition', 'grafitti_status', 'hearing_date', 'late_fee',
#                         'mailing_address_str_name', 'mailing_address_str_number', 'non_us_str_code', 'state_fee',
#                         'ticket_id',
#                         'violation_code', 'violation_street_name', 'violation_street_number', 'violation_zip_code',
#                         'violator_name',
#                         'payment_amount', 'balance_due']
#
#     # EDA Clean and split featureset
#     df.drop(garbage_features, axis=1, inplace=True)
#
#     model_features = ['violation_description', 'discount_amount', 'judgment_amount']
#
#     # ------------------------------------------ Feature engineering ------------------------------------------#
#
#     # Factorize in levels
#     df['zip_code'] = factorized(df['zip_code'])
#     df['agency_name'] = factorized(df['agency_name'])
#     df['inspector_name'] = factorized(df['inspector_name'])
#     df['violation_description'] = factorized(df['violation_description'])
#
#     # Detroit yes/no
#     df['state'] = factorized(df['state'])
#     df.loc[df['state'] != 0, 'state'] = 1
#
#     # USA yes/no
#     df.loc[df['country'] == 'USA', 'country'] = 1
#     df.loc[df['country'] != 1, 'country'] = 0
#
#     # ------------------------------------------ EDA impute NaN's ------------------------------------------#
#
#     #     null_columns = list(df.columns[df.isnull().any()])
#     #     null_objects = cat_features(df[null_columns])
#     #     null_numerics = num_features(df[null_columns])
#
#     #     # Convert date features to date and forward fill NaN's
#     #     df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
#     #     [df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]
#
#     #     # Fill NaN's with mean for categorical/numeric columns
#     #     [df.loc[:, x].fillna('', inplace=True) for x in null_objects]
#     #     [df.loc[:, x].fillna(df.loc[:, x].mean(), inplace=True) for x in null_numerics]
#     df.fillna(0, inplace=True)
#     assert df.isnull().sum().sum() == 0, 'Not all NaNs are removed'
#
#     # ------------------------------------------ Split dataset ------------------------------------------#
#
#     # Split numerical features
#     not_class2 = df['compliance'] != 2
#     X_model = df.loc[:, model_features]
#     y_model = df.loc[not_class2, 'compliance']
#
#     # ------------------------------------------ Scaling, transforming features ------------------------------------------#
#
#     # Scaling incl. class 2?
#     scaler = RobustScaler()  # StandardScaler()  # MinMaxScaler()
#     X_model = scaler.fit_transform(X_model)
#     X_model = X_model[np.array(not_class2)]
#
#     # ----------------------------------------> learn final model <---------------------------------------- #
#
#     from sklearn.ensemble import GradientBoostingClassifier
#     from sklearn.neighbors import KNeighborsClassifier
#
#     gb = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_model, y_model)
#     #     gb = GradientBoostingClassifier(learning_rate=0.05, max_depth=2, random_state=0).fit(X_model, y_model)
#
#     # ----------------------------------------> predict final model <---------------------------------------- #
#     # ----------------------------------------> predict final model <---------------------------------------- #
#
#     df = df_unseen.copy(deep=True)
#
#     #     garbage_features = ['city', 'clean_up_cost', 'disposition', 'grafitti_status', 'hearing_date', 'late_fee',
#     #      'mailing_address_str_name', 'mailing_address_str_number', 'non_us_str_code', 'state_fee', 'ticket_id',
#     #      'violation_code', 'violation_street_name', 'violation_street_number', 'violation_zip_code', 'violator_name']
#
#     # EDA Clean and split featureset
#     #     df.drop(garbage_features, axis=1, inplace=True)
#
#     model_features = ['violation_description', 'discount_amount', 'judgment_amount']
#     #     df = df.loc[:, model_features]
#
#     # ------------------------------------------ Feature engineering ------------------------------------------#
#
#     # Factorize in levels
#     #     df['zip_code'] = factorized(df['zip_code'])
#     #     df['agency_name'] = factorized(df['agency_name'])
#     #     df['inspector_name'] = factorized(df['inspector_name'])
#     df['violation_description'] = factorized(df['violation_description'])
#
#     #     # Detroit yes/no
#     #     df['state'] = factorized(df['state'])
#     #     df.loc[df['state'] != 0, 'state'] = 1
#
#     #     # USA yes/no
#     #     df.loc[df['country'] == 'USA', 'country'] = 1
#     #     df.loc[df['country'] != 1, 'country'] = 0
#
#     # ------------------------------------------ EDA impute NaN's ------------------------------------------#
#
#     #     null_columns = list(df.columns[df.isnull().any()])
#     #     null_objects = cat_features(df[null_columns])
#     #     null_numerics = num_features(df[null_columns])
#
#     # Convert date features to date and forward fill NaN's
#     #     df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
#     #     [df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]
#
#     # Fill NaN's with mean for categorical/numeric columns
#     #     [df.loc[:, x].fillna('', inplace=True) for x in null_objects]
#     #     [df.loc[:, x].fillna(df.loc[:, x].mean(), inplace=True) for x in null_numerics]
#     df.fillna(0, inplace=True)
#     assert df.isnull().sum().sum() == 0, 'Not all NaNs are removed'
#
#     # ------------------------------------------ Scaling, transforming features ------------------------------------------#
#
#     df_model = df.loc[:, model_features]
#     scaler = RobustScaler()
#     X_unseen = scaler.fit_transform(df_model)
#
#     predicted = gb.predict(X_unseen)
#     pred_series = pd.DataFrame(predicted)
#     pred_series['ticket_id'] = df_unseen['ticket_id']
#     pred_series.set_index('ticket_id', inplace=True)
#
#     return pred_series
#
# ######################
# # --------------------------------------------->  <-------------------------------------------- #
#
# import pandas as pd
# import numpy as np
#
#
# def blight_model():
#
#     from sklearn.preprocessing import RobustScaler
#     from sklearn.ensemble import GradientBoostingClassifier
#     from sklearn.neighbors import KNeighborsClassifier
#
#     def cat_features(df):
#         return df.select_dtypes(include=['object', 'category']).columns
#
#     def factorized(df):
#         if isinstance(df, pd.Series):
#             return df.factorize()[0]
#         return df.loc[:, cat_features(df)].apply(lambda x: x.factorize()[0])
#
#     # Import
#     set_types = {'compliance': object}
#     df_train = pd.read_csv('train.csv', encoding='latin-1', low_memory=False, dtype=set_types)
#     df = df_train.copy(deep=True)
#     df['compliance'] = df['compliance'].fillna(2).replace({'0.0': 0, '1.0': 1}).astype('category')
#
#     # Feature selection
#     model_features = ['violation_description', 'discount_amount', 'judgment_amount']
#
#     # Engineer
#     df['violation_description'] = factorized(df['violation_description'])
#     df.fillna(0, inplace=True)
#     assert df.isnull().sum().sum() == 0, 'Not all NaNs are removed'
#
#     # Split numerical features
#     not_class2 = df['compliance'] != 2
#     X_model = df.loc[:, model_features]
#     y_model = df.loc[not_class2, 'compliance']
#
#     # Scaling incl. class 2?
#     scaler = RobustScaler()  # StandardScaler()  # MinMaxScaler()
#     X_model = scaler.fit_transform(X_model)
#     X_model = X_model[np.array(not_class2)]
#
#     # Train
#     fit = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_model, y_model)
#     #     fit = GradientBoostingClassifier(learning_rate=0.05, max_depth=2, random_state=0).fit(X_model, y_model)
#
#     # Set df
#     df_unseen = pd.read_csv('test.csv', encoding='latin-1', low_memory=False)
#     df = df_unseen.copy(deep=True)
#
#     # Engineer
#     df['violation_description'] = factorized(df['violation_description'])
#     df.fillna(0, inplace=True)
#     assert df.isnull().sum().sum() == 0, 'Not all NaNs are removed'
#
#     # Select and scale
#     df_model = df.loc[:, model_features]
#     scaler = RobustScaler()
#     X_unseen = scaler.fit_transform(df_model)
#
#     # Predict
#     predicted = fit.predict(X_unseen)
#     pred_series = pd.DataFrame(predicted)
#     pred_series['ticket_id'] = df_unseen['ticket_id']
#     pred_series.set_index('ticket_id', inplace=True)
#
#     return pred_series

# --------------------------------------------->  <-------------------------------------------- #

def blight_model():

    import numpy as np
    import pandas as pd

    def date_features(df):
        return df.columns.str.extractall(r'(.*date.*)')[0].values.tolist()

    def clip_outliers(values, p=99):
        """clip at 1.5 IQR"""
        min = np.percentile(values, 100 - p)
        max = np.percentile(values, p)
        return np.clip(values, min, max)

    def numerize_code(df, feature, replace=0):
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
        nums = df[feature].fillna(replace).astype('int64')
        return clip_outliers(nums)

    def top_cat(df_, feature, top=10):
        """Replace top 10 most frequent labels with 0-9 and rest with 10"""
        alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        labels = alphabet[:top]
        other = alphabet[top + 1]
        top_violation_codes = df_.groupby(feature)[feature].count().sort_values(ascending=False).head(
            top).index.tolist()
        map_values = {k: l for k, l in (zip(top_violation_codes, labels))}  # [::-1]
        key_others = set(map_values.keys()) ^ (set(df_.loc[:, feature].values))
        map_others = {k: other for k in key_others}
        map_all = {**map_others, **map_values}
        df_.loc[:, feature] = df_.loc[:, feature].replace(map_all).astype('category')
        return df_

    # ------------------------------------------ Import datasets

    # Import
    set_types = {'compliance': object}
    df_train = pd.read_csv('train.csv', encoding='latin-1', low_memory=False, dtype=set_types)
    df_unseen = pd.read_csv('test.csv', encoding='latin-1', low_memory=False, dtype=set_types)
    df_zip = pd.read_csv('zipcode.csv')

    # ---------------------------------------------> Merge datasets on zip_code

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
    M = df_train.loc[:, 'zip_code'].isin(df_zip.loc[:, 'zip'])  # symmetric_difference

    # Most frequent zip_code for imputing
    top_zips = df_train.loc[:, 'zip_code'].value_counts().head(25)
    zip_samples = np.random.choice(top_zips.index, len(df_train.loc[~M, 'zip_code']))
    df_train.loc[~M, 'zip_code'] = zip_samples

    # Merge
    df_merged = pd.merge(df_train, df_zip, how='left', left_on='zip_code', right_on='zip')
    df_merged.loc(1)['zip'] = numerize_code(df_merged, 'zip', 99)
    # set(df_merged.loc[:, 'zip_code']).symmetric_difference(set(df_merged.loc[:, 'zip']))

    # Generalize working phase
    df = df_merged.copy(deep=True)

    # ------------------------------------------ Feature Selection

    # Set y(target) to proper category type
    df['compliance'] = df['compliance'].fillna(2).replace({'0.0': 0, '1.0': 1}).astype('category')
    # Features
    train_only_features = ['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status']
    redundant_features = ['ticket_id', 'violation_street_name', 'violation_street_number', 'violation_zip_code',
                          'violation_description', 'mailing_address_str_name', 'mailing_address_str_number',
                          'non_us_str_code', 'grafitti_status', 'dst', 'timezone', 'compliance_detail', 'agency_name',
                          'inspector_name', 'violator_name', 'city_x', 'state_x', 'city_y', 'state_y', 'country', 'zip',
                          'zip_code', 'discount_amount', 'state_fee', 'clean_up_cost', 'discount_amount', 'admin_fee']
    garbage_features = set(train_only_features) | set(redundant_features)

    df.drop(garbage_features, axis=1, inplace=True)
    df = df.dropna(axis=0)

    # ---------------------------------------------> EDA & Feature Engineering

    # Mask (minor) class 2 for plotting
    trgt = 'compliance'
    M = (df.loc[:, trgt] == 1)

    # convert all dates features
    df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
    _ = [df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]

    # ---------------------------------------------> latlong <------------------------------------------------------------ #
    # Geo, numerical
    lat = df.loc[:, 'latitude']
    long = df.loc[:, 'longitude']

    # ---------------------------------------------> 'ticket_issued_date' <----------------------------------------------- #
    # categorical - t_month, t_day_of_week
    ftr = 'ticket_issued_date'
    df['t_month'] = df.loc(1)[ftr].dt.month
    df['t_day_of_week'] = df.loc(1)[ftr].dt.dayofweek
    df.drop(ftr, axis=1, inplace=True)

    # ---------------------------------------------> 'hearing_date' <----------------------------------------------------- #
    # categorical - h_month, h_day_of_week
    ftr = 'hearing_date'
    df['h_month'] = df.loc(1)[ftr].dt.month
    df['h_day_of_week'] = df.loc(1)[ftr].dt.dayofweek
    df.drop(ftr, axis=1, inplace=True)

    # ---------------------------------------------> 'violation_code' <--------------------------------------------------- #
    # categorical
    ftr = 'violation_code'
    # replace values of feature
    top_cat(df, ftr, 10)

    # ---------------------------------------------> 'fine_amount' <------------------------------------------------------ #
    # binary
    ftr = 'fine_amount'
    thres_min, thres_max = 20, 251
    df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
    df[ftr] = df['thres'].astype('int')

    # ---------------------------------------------> 'judgment_amount' <-------------------------------------------------- #
    # binary
    ftr = 'judgment_amount'
    thres_min, thres_max = 50, 300
    df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
    df[ftr] = df['thres'].astype('int')

    # ---------------------------------------------> 'late_fee' <--------------------------------------------------------- #
    # binary
    ftr = 'late_fee'
    thres_min, thres_max = -0.1, 1
    df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
    df[ftr] = df['thres'].astype('int')

    # ---------------------------------------------> 'disposition' <------------------------------------------------------ #
    # binary
    ftr = 'disposition'
    # Replace values of feature
    top_cat(df, ftr, top=1)

    # ---------------------------------------------> Last Feature Cleaning

    df.drop('thres', axis=1, inplace=True)
    df = pd.get_dummies(df, columns=['violation_code', 'disposition'], drop_first=True)

    # ------------------------------------------ Final dataset

    assert df.isnull().sum().sum() == 0, 'Not all NaNs are removed'
    train_features = set(df.columns.tolist()) ^ set(['compliance'])
    model_X = df.loc[:, train_features]
    model_y = df.loc[:, 'compliance']

    # ---------------------------------------------> Import sklearn packages

    # Import preprocessing, selection and metrics
    from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
    from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

    # Import classifiers
    from sklearn.dummy import DummyClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC

    # ---------------------------------------------> Split, transform, helper dataset

    def split_transform(*args, phase='train'):
        X, y = args
        not_class2 = (y != 2)
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
            X_model = scaler.fit_transform(X)  # TODO error
            X_model = X_model[np.array(not_class2)]  # syntax! [np.array[mask]]
            y_model = y[np.array(not_class2)]
            return X_model, y_model
        if phase == 'predict':
            X_unseen = scaler.fit_transform(X)
            return X_unseen

    # ---------------------------------------------> Final Model Fit

    X_model, y_model = split_transform(model_X, model_y, phase='model')
    nb = GaussianNB().fit(X_model, y_model)

    # ---------------------------------------------> Test Final Model

    test_features = ['fine_amount', 'late_fee', 'judgment_amount', 'ticket_issued_date', 'hearing_date',
                     'violation_code',
                     'zip_code', 'disposition']
    df = df_unseen.loc[:, set(test_features)]

    # find outliers in zip_code
    df.loc[:, 'zip_code'] = numerize_code(df, 'zip_code', 99).astype('i')
    M = df.loc[:, 'zip_code'].isin(df_zip.loc[:, 'zip'])  # symmetric_difference
    # df.loc[~M, 'zip_code'].value_counts().head(20)

    # Most frequent zip_code for imputing
    top_zips = df.loc[:, 'zip_code'].value_counts().head(25)
    zip_samples = np.random.choice(top_zips.index, len(df.loc[~M, 'zip_code']))
    df.loc[~M, 'zip_code'] = zip_samples

    # Merge on zip_code - add latlong
    df_merged = pd.merge(df, df_zip, how='left', left_on='zip_code', right_on='zip')
    df_merged.loc(1)['zip'] = numerize_code(df_merged, 'zip', 99)
    df = df_merged

    # convert all dates
    df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
    _ = [df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]

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

    df.drop(['city', 'dst', 'state', 'timezone', 'zip', 'zip_code', 'thres'], axis=1, inplace=True)

    df = pd.get_dummies(df, columns=['violation_code', 'disposition'], drop_first=True)

    assert df.isnull().sum().sum() == 0, 'Not all NaNs are removed'
    assert set(df.columns.tolist()) == set(train_features), 'Features train and test is not same'
    test_features = set(df.columns.tolist())
    model_X = df.loc[:, test_features]


    X_unseen = split_transform(model_X, None, phase='predict')

    predicted = nb.predict(X_unseen)
    pred_series = pd.DataFrame(predicted)
    pred_series['ticket_id'] = df_unseen['ticket_id']
    pred_series.set_index('ticket_id', inplace=True)

    # pred_series = None

    return pred_series

blight_model()

# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------> End <--------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


# --------------------------------------------->  <-------------------------------------------- #

#####
# import numpy as np
# a = np.arange(6)
#
# b = [0, 5, 2, 3, 4, 5]
# max_mask = [x == max(b) for x in b]
