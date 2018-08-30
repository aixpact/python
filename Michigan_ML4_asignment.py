# import numpy as np
# import pandas as pd
#
# def blight_model0():
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
#     def top_cat(df_, feature, top=10):
#         """Replace top 10 most frequent labels with 0-9 and rest with 10"""
#         alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
#         labels = alphabet[:top]
#         other = alphabet[top + 1]
#         top_violation_codes = df_.groupby(feature)[feature].count().sort_values(ascending=False).head(
#             top).index.tolist()
#         map_values = {k: l for k, l in (zip(top_violation_codes, labels))}  # [::-1]
#         key_others = set(map_values.keys()) ^ (set(df_.loc[:, feature].values))
#         map_others = {k: other for k in key_others}
#         map_all = {**map_others, **map_values}
#         df_.loc[:, feature] = df_.loc[:, feature].replace(map_all).astype('category')
#         return df_
#
#     # ------------------------------------------ Import datasets
#
#     # Import
#     set_types = {'compliance': object}
#     df_train = pd.read_csv('train.csv', encoding='latin-1', low_memory=False, dtype=set_types)
#     df_unseen = pd.read_csv('test.csv', encoding='latin-1', low_memory=False, dtype=set_types)
#     df_addresses = pd.read_csv('addresses.csv')
#     df_latlons = pd.read_csv('latlons.csv')
#
#     # ---------------------------------------------> Merge datasets on zip_code
#     def merge_df(df_, df_latlon, df_address):
#         df_zip = pd.merge(df_latlon, df_address, how='left', left_on='address', right_on='address')
#         df_merge = pd.merge(df_, df_zip, how='left', left_on='ticket_id', right_on='ticket_id')
#         return df_merge
#
#     df_train = merge_df(df_train, df_latlons, df_addresses)
#     df_test = merge_df(df_unseen, df_latlons, df_addresses)
#
#     # ------------------------------------------ Feature Selection
#
#     # Drop features
#     garbage_features = ['ticket_id', 'violation_street_name', 'violation_street_number', 'violation_zip_code',
#                           'violation_description', 'mailing_address_str_name', 'mailing_address_str_number',
#                           'non_us_str_code', 'grafitti_status', 'agency_name', 'state', 'city', 'address',
#                           'inspector_name', 'violator_name', 'country', 'zip_code', 'discount_amount', 'state_fee',
#                           'clean_up_cost', 'discount_amount', 'admin_fee']
#     df_test.drop(garbage_features, axis=1, inplace=True)
#     df_train.drop(garbage_features, axis=1, inplace=True)
#
#     train_only_features = ['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status',
#                            'compliance_detail']
#     df_train.drop(train_only_features, axis=1, inplace=True)
#
#
#     # NaN imputation - train
#     thres = 0.001
#     nan_features = list(df_train.columns[df_train.isnull().mean() > thres])
#     # assert nan_features == [], '{} need NaN imputation'.format(nan_features)
#     df_train.dropna(axis=0, inplace=True)
#     assert df_train.isnull().sum().sum() == 0, 'Not all NaNs are removed'
#
#     # NaN imputation - test
#     thres = 0
#     nan_features = list(df_test.columns[df_test.isnull().mean() > thres])
#     # assert nan_features == [], '{} need NaN imputation'.format(nan_features)
#     for ftr in nan_features:
#         df_test.loc(1)[ftr] = df_test.loc(1)[ftr].fillna(method='pad')
#     assert df_test.isnull().sum().sum() == 0, 'Not all NaNs are removed'
#
#     # Sanity check
#     # for df in [df_train, df_test]:
#     #     df_train.info()
#     #     df_test.info()
#
#     #
#     df_train['compliance'] = df_train['compliance'].fillna(2).replace({'0.0': 0, '1.0': 1}).astype('category')
#     M = (df_train.loc[:, 'compliance'] == 0) | (df_train.loc[:, 'compliance'] == 1)
#     df_train = df_train.loc[M, :].copy()
#     assert df_train.is_copy == None, 'Check if use proper copy'
#
#     # ---------------------------------------------> EDA & Feature Engineering
#
#     # convert all dates features
#     for df in [df_train, df_test]:
#         df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
#         _ = [df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]
#
#     # ---------------------------------------------> 'ticket_issued_date' <----------------------------------------------- #
#     # categorical - t_month, t_day_of_week
#     ftr = 'ticket_issued_date'
#     for df in [df_train, df_test]:
#         df['t_month'] = df.loc(1)[ftr].dt.month
#         df['t_day_of_week'] = df.loc(1)[ftr].dt.dayofweek
#
#
#     # ---------------------------------------------> 'hearing_date' <----------------------------------------------------- #
#     # categorical - h_month, h_day_of_week
#     ftr = 'hearing_date'
#     for df in [df_train, df_test]:
#         df['h_month'] = df.loc(1)[ftr].dt.month
#         df['h_day_of_week'] = df.loc(1)[ftr].dt.dayofweek
#
#     # ---------------------------------------------> 'elapsed_days' <-------------------------------------------- #
#     ftr_h = 'hearing_date'
#     ftr_t = 'ticket_issued_date'
#     for df in [df_train, df_test]:
#         df['elapsed_days'] = clip_outliers(pd.to_numeric(df.loc(1)[ftr_h] - df.loc(1)[ftr_t])/1e14).astype('i')
#         df.drop(ftr_t, axis=1, inplace=True)
#         df.drop(ftr_h, axis=1, inplace=True)
#
#     # ---------------------------------------------> 'violation_code' <--------------------------------------------------- #
#     # categorical
#     ftr = 'violation_code'
#     for df in [df_train, df_test]:
#         # replace values of feature
#         top_cat(df, ftr, 10)
#
#     # ---------------------------------------------> 'fine_amount' <------------------------------------------------------ #
#     # binary
#     ftr = 'fine_amount'
#     for df in [df_train, df_test]:
#         thres_min, thres_max = 20, 251
#         df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
#         df[ftr] = df['thres'].astype('int')
#
#     # ---------------------------------------------> 'judgment_amount' <-------------------------------------------------- #
#     # binary
#     ftr = 'judgment_amount'
#     for df in [df_train, df_test]:
#         thres_min, thres_max = 50, 300
#         df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
#         df[ftr] = df['thres'].astype('int')
#
#     # ---------------------------------------------> 'late_fee' <--------------------------------------------------------- #
#     # binary
#     ftr = 'late_fee'
#     for df in [df_train, df_test]:
#         thres_min, thres_max = -0.1, 1
#         df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
#         df[ftr] = df['thres'].astype('int')
#
#     # ---------------------------------------------> 'disposition' <------------------------------------------------------ #
#     # binary
#     ftr = 'disposition'
#     for df in [df_train, df_test]:
#         # Replace values of feature
#         top_cat(df, ftr, top=1)
#
#     # ---------------------------------------------> Last Feature Cleaning
#
#     for df in [df_train, df_test]:
#         try:
#             df.drop('thres', axis=1, inplace=True)
#         except:
#             continue
#
#     df_train = pd.get_dummies(df_train, columns=['violation_code', 'disposition'], drop_first=True)
#     df_test = pd.get_dummies(df_test, columns=['violation_code', 'disposition'], drop_first=True)
#
#     # Final sanity check
#     # for df in [df_train, df_test]:
#         # df_train.info()
#         # df_test.info()
#
#     # ------------------------------------------ Final dataset
#
#
#     # train_features = set(df.columns.tolist()) ^ set(['compliance'])
#     model_y = df_train.pop('compliance')
#     model_X = df_train
#
#     # ---------------------------------------------> Import sklearn packages
#
#     # Import preprocessing, selection and metrics
#     from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
#     from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
#
#     # Import classifiers
#     from sklearn.dummy import DummyClassifier
#     from sklearn.tree import DecisionTreeClassifier
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.naive_bayes import GaussianNB
#     from sklearn.ensemble import GradientBoostingClassifier
#     from sklearn.neighbors import KNeighborsClassifier
#     from sklearn.svm import LinearSVC
#
#     # ---------------------------------------------> Split, transform, helper dataset
#
#     def split_transform(*args, phase='train'):
#         X, y = args
#         not_class2 = (y != 2)
#         scaler = RobustScaler()  # StandardScaler()  # MinMaxScaler()
#         if phase == 'train':
#             X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)
#             # Mulit class to binary
#             X_train, X_test = X_train[np.array(y_train != 2)], X_test[np.array(y_test != 2)]
#             y_train, y_test = y_train[np.array(y_train != 2)], y_test[np.array(y_test != 2)]
#             return X_train, X_test, y_train, y_test
#         if phase == 'model':
#             X_model = scaler.fit_transform(X)
#             X_model = X_model[np.array(not_class2)]  # syntax! [np.array[mask]]
#             y_model = y[np.array(not_class2)]
#             return X_model, y_model
#         if phase == 'predict':
#             X_unseen = scaler.fit_transform(X)
#             return X_unseen
#
#     # ---------------------------------------------> Final Model Fit
#
#     X_model, y_model = split_transform(model_X, model_y, phase='model')
#     nb = GaussianNB().fit(X_model, y_model)
#
#     # ---------------------------------------------> Test Final Model
#
#     X_unseen = split_transform(df_test, None, phase='predict')
#
#     predicted = nb.predict(X_unseen)
#     pred_series = pd.DataFrame(predicted)
#     pred_series['ticket_id'] = df_unseen['ticket_id']
#     pred_series.set_index('ticket_id', inplace=True)
#
#     return pred_series
#
# blight_model0()
#
# # -------------------------------------------------------------------------------------------------------------------- #
# # ----------------------------------------> End <--------------------------------------------------------------------- #
# # -------------------------------------------------------------------------------------------------------------------- #
#
#
# # --------------------------------------------->  <-------------------------------------------- #
#
# #####
# # import numpy as np
# # a = np.arange(6)
# #
# # b = [0, 5, 2, 3, 4, 5]
# # max_mask = [x == max(b) for x in b]
#
# # --------------------------------------------->  <-------------------------------------------- #
# # --------------------------------------------->  <-------------------------------------------- #
#
# import numpy as np
# import pandas as pd
#
#
# def blight_model1():
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
#     def top_cat(df_, feature, top=10):
#         """Replace top 10 most frequent labels with 0-9 and rest with 10"""
#         alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
#         labels = alphabet[:top]
#         other = alphabet[top + 1]
#         top_violation_codes = df_.groupby(feature)[feature].count().sort_values(ascending=False).head(
#             top).index.tolist()
#         map_values = {k: l for k, l in (zip(top_violation_codes, labels))}  # [::-1]
#         key_others = set(map_values.keys()) ^ (set(df_.loc[:, feature].values))
#         map_others = {k: other for k in key_others}
#         map_all = {**map_others, **map_values}
#         df_.loc[:, feature] = df_.loc[:, feature].replace(map_all).astype('category')
#         return df_
#
#     # ------------------------------------------ Import datasets
#
#     # Import
#     set_types = {'compliance': object}
#     df_train = pd.read_csv('train.csv', encoding='latin-1', low_memory=False, dtype=set_types)
#     df_unseen = pd.read_csv('test.csv', encoding='latin-1', low_memory=False, dtype=set_types)
#     df_addresses = pd.read_csv('addresses.csv')
#     df_latlons = pd.read_csv('latlons.csv')
#
#     # ---------------------------------------------> Merge datasets on zip_code
#     def merge_df(df_, df_latlon, df_address):
#         df_zip = pd.merge(df_latlon, df_address, how='left', left_on='address', right_on='address')
#         df_merge = pd.merge(df_, df_zip, how='left', left_on='ticket_id', right_on='ticket_id')
#         return df_merge
#
#     df_train = merge_df(df_train, df_latlons, df_addresses)
#     df_test = merge_df(df_unseen, df_latlons, df_addresses)
#
#     # ------------------------------------------ Feature Selection
#
#     # Drop features
#     garbage_features = ['ticket_id', 'violation_street_name', 'violation_street_number', 'violation_zip_code',
#                         'violation_description', 'mailing_address_str_name', 'mailing_address_str_number',
#                         'non_us_str_code', 'grafitti_status', 'agency_name', 'state', 'city', 'address',
#                         'inspector_name', 'violator_name', 'country', 'zip_code', 'discount_amount', 'state_fee',
#                         'clean_up_cost', 'discount_amount', 'admin_fee']
#     df_test.drop(garbage_features, axis=1, inplace=True)
#     df_train.drop(garbage_features, axis=1, inplace=True)
#
#     train_only_features = ['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status',
#                            'compliance_detail']
#     df_train.drop(train_only_features, axis=1, inplace=True)
#
#     # NaN imputation - train
#     thres_train = 0.001
#     nans_train = list(df_train.columns[df_train.isnull().mean() > thres_train])
#     # assert nan_features == [], '{} need NaN imputation'.format(nan_features)
#     df_train.dropna(axis=0, inplace=True)
#     assert df_train.isnull().sum().sum() == 0, 'Not all NaNs are removed'
#
#     # NaN imputation - test
#     thres_test = 0
#     nans_test = list(df_test.columns[df_test.isnull().mean() > thres_test])
#     # assert nan_features == [], '{} need NaN imputation'.format(nan_features)
#     for ftr in nans_test:
#         df_test.loc[:, ftr] = df_test.loc[:, ftr].fillna(method='pad')
#     assert df_test.isnull().sum().sum() == 0, 'Not all NaNs are removed'
#
#     print(df_train.columns)
#
#     # Sanity check
#     for df in [df_train, df_test]:
#         df_train.info()
#         df_test.info()
#         print(df.columns)
#
#     #
#     df_train['compliance'] = df_train['compliance'].fillna(2).replace({'0.0': 0, '1.0': 1}).astype('category')
#     M = (df_train.loc[:, 'compliance'] == 0) | (df_train.loc[:, 'compliance'] == 1)
#     df_train = df_train.loc[M, :].copy()
#     assert df_train.is_copy == None, 'Check if use proper copy'
#
#     # ---------------------------------------------> EDA & Feature Engineering
#
#     # convert all dates features
#     for df in [df_train, df_test]:
#         df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
#         _ = [df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]
#
#     # ---------------------------------------------> 'ticket_issued_date' <----------------------------------------------- #
#     # categorical - t_month, t_day_of_week
#     ftr = 'ticket_issued_date'
#     for df in [df_train, df_test]:
#         df['t_month'] = df.loc[:, ftr].dt.month
#         df['t_day_of_week'] = df.loc[:, ftr].dt.dayofweek
#
#     # ---------------------------------------------> 'hearing_date' <----------------------------------------------------- #
#     # categorical - h_month, h_day_of_week
#     ftr = 'hearing_date'
#     for df in [df_train, df_test]:
#         df['h_month'] = df.loc[:, ftr].dt.month
#         df['h_day_of_week'] = df.loc[:, ftr].dt.dayofweek
#
#     # ---------------------------------------------> 'elapsed_days' <-------------------------------------------- #
#     ftr_h = 'hearing_date'
#     ftr_t = 'ticket_issued_date'
#     for df in [df_train, df_test]:
#         df['elapsed_days'] = clip_outliers(pd.to_numeric(df.loc[:, ftr_h] - df.loc[:, ftr_t]) / 1e14).astype('i')
#         df.drop(ftr_t, axis=1, inplace=True)
#         df.drop(ftr_h, axis=1, inplace=True)
#
#     # ---------------------------------------------> 'violation_code' <--------------------------------------------------- #
#     # categorical
#     ftr = 'violation_code'
#     for df in [df_train, df_test]:
#         # replace values of feature
#         top_cat(df, ftr, 10)
#
#     # ---------------------------------------------> 'fine_amount' <------------------------------------------------------ #
#     # binary
#     ftr = 'fine_amount'
#     for df in [df_train, df_test]:
#         thres_min, thres_max = 20, 251
#         df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
#         df[ftr] = df['thres'].astype('int')
#
#     # ---------------------------------------------> 'judgment_amount' <-------------------------------------------------- #
#     # binary
#     ftr = 'judgment_amount'
#     for df in [df_train, df_test]:
#         thres_min, thres_max = 50, 300
#         df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
#         df[ftr] = df['thres'].astype('int')
#
#     # ---------------------------------------------> 'late_fee' <--------------------------------------------------------- #
#     # binary
#     ftr = 'late_fee'
#     for df in [df_train, df_test]:
#         thres_min, thres_max = -0.1, 1
#         df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
#         df[ftr] = df['thres'].astype('int')
#
#     # ---------------------------------------------> 'disposition' <------------------------------------------------------ #
#     # binary
#     ftr = 'disposition'
#     for df in [df_train, df_test]:
#         # Replace values of feature
#         top_cat(df, ftr, top=1)
#
#     # ---------------------------------------------> Last Feature Cleaning
#
#     for df in [df_train, df_test]:
#         try:
#             df.drop('thres', axis=1, inplace=True)
#         except:
#             continue
#
#     df_train = pd.get_dummies(df_train, columns=['violation_code', 'disposition'], drop_first=True)
#     df_test = pd.get_dummies(df_test, columns=['violation_code', 'disposition'], drop_first=True)
#
#     # Final sanity check
#     # for df in [df_train, df_test]:
#     # df_train.info()
#     # df_test.info()
#
#     # ------------------------------------------ Final dataset
#
#     # train_features = set(df.columns.tolist()) ^ set(['compliance'])
#     model_y = df_train.pop('compliance')
#     model_X = df_train
#
#     # ---------------------------------------------> Import sklearn packages
#
#     # Import preprocessing, selection and metrics
#     from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
#     from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
#
#     # Import classifiers
#     from sklearn.dummy import DummyClassifier
#     from sklearn.tree import DecisionTreeClassifier
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.naive_bayes import GaussianNB
#     from sklearn.ensemble import GradientBoostingClassifier
#     from sklearn.neighbors import KNeighborsClassifier
#     from sklearn.svm import LinearSVC
#
#     # ---------------------------------------------> Split, transform, helper dataset
#
#     def split_transform(*args, phase='train'):
#         X, y = args
#         not_class2 = (y != 2)
#         scaler = RobustScaler()  # StandardScaler()  # MinMaxScaler()
#         if phase == 'train':
#             X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)
#             # Mulit class to binary
#             X_train, X_test = X_train[np.array(y_train != 2)], X_test[np.array(y_test != 2)]
#             y_train, y_test = y_train[np.array(y_train != 2)], y_test[np.array(y_test != 2)]
#             return X_train, X_test, y_train, y_test
#         if phase == 'model':
#             X_model = scaler.fit_transform(X)
#             X_model = X_model[np.array(not_class2)]  # syntax! [np.array[mask]]
#             y_model = y[np.array(not_class2)]
#             return X_model, y_model
#         if phase == 'predict':
#             X_unseen = scaler.fit_transform(X)
#             return X_unseen
#
#     # ---------------------------------------------> Final Model Fit
#
#     X_model, y_model = split_transform(model_X, model_y, phase='model')
#     nb = GaussianNB().fit(X_model, y_model)
#
#     # ---------------------------------------------> Test Final Model
#
#     X_unseen = split_transform(df_test, None, phase='predict')
#
#     predicted = nb.predict(X_unseen)
#     pred_series = pd.DataFrame(predicted)
#     pred_series['ticket_id'] = df_unseen['ticket_id']
#     pred_series.set_index('ticket_id', inplace=True)
#
#     pred_series = None
#
#     return pred_series
#
# # --------------------------------------------->  <-------------------------------------------- #

import numpy as np
import pandas as pd


def blight_model_():
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
    df_addresses = pd.read_csv('addresses.csv')
    df_latlons = pd.read_csv('latlons.csv')

    # ---------------------------------------------> Merge datasets on zip_code
    def merge_df(df_, df_latlon, df_address):
        df_zip = pd.merge(df_latlon, df_address, how='left', left_on='address', right_on='address')
        df_merge = pd.merge(df_, df_zip, how='left', left_on='ticket_id', right_on='ticket_id')
        return df_merge

    df_train = merge_df(df_train, df_latlons, df_addresses)
    df_test = merge_df(df_unseen, df_latlons, df_addresses)

    # ------------------------------------------ Feature Selection


    # Drop features
    garbage_features = ['ticket_id', 'violation_street_name', 'violation_street_number', 'violation_zip_code',
                        'violation_code', 'mailing_address_str_name', 'mailing_address_str_number',
                        'non_us_str_code', 'grafitti_status', 'state', 'city', 'address', 'agency_name',
                        'inspector_name', 'violator_name', 'country', 'zip_code', 'discount_amount', 'state_fee',
                        'clean_up_cost', 'discount_amount', 'admin_fee']
    df_test.drop(garbage_features, axis=1, inplace=True)
    df_train.drop(garbage_features, axis=1, inplace=True)

    train_only_features = ['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status',
                           'compliance_detail']
    df_train.drop(train_only_features, axis=1, inplace=True)

    # NaN imputation - train
    thres_train = 0.001
    nans_train = list(df_train.columns[df_train.isnull().mean() > thres_train])
    # assert nan_features == [], '{} need NaN imputation'.format(nan_features)
    df_train.dropna(axis=0, inplace=True)
    assert df_train.isnull().sum().sum() == 0, 'Not all NaNs are removed'

    # NaN imputation - test
    thres_test = 0
    nans_test = list(df_test.columns[df_test.isnull().mean() > thres_test])
    # assert nan_features == [], '{} need NaN imputation'.format(nan_features)
    for ftr in nans_test:
        df_test.loc[:, ftr] = df_test.loc[:, ftr].fillna(method='pad')
    assert df_test.isnull().sum().sum() == 0, 'Not all NaNs are removed'

    #
    df_train['compliance'] = df_train['compliance'].fillna(2).replace({'0.0': 0, '1.0': 1}).astype('category')
    M = (df_train.loc[:, 'compliance'] == 0) | (df_train.loc[:, 'compliance'] == 1)
    df_train = df_train.loc[M, :].copy()
    assert df_train.is_copy == None, 'Check if use proper copy'

    # ---------------------------------------------> EDA & Feature Engineering

    # List of categorical features to convert to dummy vars
    dummies = []

    # convert all dates features
    for df in [df_train, df_test]:
        df.loc[:, date_features(df)] = df.loc[:, date_features].apply(pd.to_datetime)
        _ = [df.loc[:, x].fillna(method='ffill', inplace=True) for x in date_features(df)]

    # ---------------------------------------------> 'ticket_issued_date'
    # categorical - t_month, t_day_of_week
    ftr = 'ticket_issued_date'
    # for df in [df_train, df_test]:
    #     df['t_year'] = df.loc[:, ftr].dt.year
    #     df['t_month'] = df.loc[:, ftr].dt.month
    #     df['t_day_of_week'] = df.loc[:, ftr].dt.dayofweek

    # ---------------------------------------------> 'hearing_date'
    # categorical - h_month, h_day_of_week
    ftr = 'hearing_date'
    # for df in [df_train, df_test]:
        # df['t_year'] = df.loc[:, ftr].dt.year
        # df['h_month'] = df.loc[:, ftr].dt.month
        # df['h_day_of_week'] = df.loc[:, ftr].dt.dayofweek

    # ---------------------------------------------> 'elapsed_days'
    ftr_h = 'hearing_date'
    ftr_t = 'ticket_issued_date'
    for df in [df_train, df_test]:
        # df['elapsed_days'] = clip_outliers(pd.to_numeric(df.loc[:, ftr_h] - df.loc[:, ftr_t]) / 1e14).astype('i')
        df.drop(ftr_t, axis=1, inplace=True)
        df.drop(ftr_h, axis=1, inplace=True)

    # ---------------------------------------------> 'violation_code'
    # categorical
    ftr = 'violation_description'
    for df in [df_train, df_test]:
        # replace values of feature
        top_cat(df, ftr, 4)
    dummies.append(ftr)

    # ---------------------------------------------> TRY
    # categorical
    # ftr =
    # for df in [df_train, df_test]:
    #     # replace values of feature
    #     top_cat(df, ftr, 3)
    # dummies.append(ftr)

    # ---------------------------------------------> 'fine_amount'
    # binary
    ftr = 'fine_amount'
    # for df in [df_train, df_test]:
    #     thres_min, thres_max = 20, 250
    #     df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
    #     df[ftr] = df['thres'].astype('int')

    for df in [df_train, df_test]:
        # Replace values of feature
        top_cat(df, ftr, top=4)
    dummies.append(ftr)

    # ---------------------------------------------> 'judgment_amount'
    # binary
    ftr = 'judgment_amount'
    # for df in [df_train, df_test]:
    #     thres_min, thres_max = 50, 300
    #     df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
    #     df[ftr] = df['thres'].astype('int')

    for df in [df_train, df_test]:
        # Replace values of feature
        top_cat(df, ftr, top=3)
    dummies.append(ftr)

    # ---------------------------------------------> 'late_fee'
    # binary
    ftr = 'late_fee'
    for df in [df_train, df_test]:
        thres_min, thres_max = -0.1, 1
        df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
        df[ftr] = df['thres'].astype('int')

    # for df in [df_train, df_test]:
    #     # Replace values of feature
    #     top_cat(df, ftr, top=2)
    # dummies.append(ftr)

    # ---------------------------------------------> 'disposition'
    # binary
    ftr = 'disposition'
    for df in [df_train, df_test]:
        # Replace values of feature
        top_cat(df, ftr, top=2)
    dummies.append(ftr)


    # ---------------------------------------------> Last Feature Cleaning

    for df in [df_train, df_test]:
        try:
            df.drop('thres', axis=1, inplace=True)
        except:
            continue

    df_train = pd.get_dummies(df_train, columns=dummies, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=dummies, drop_first=True)

    # ------------------------------------------ Final dataset

    # train_features = set(df.columns.tolist()) ^ set(['compliance'])
    model_y = df_train.pop('compliance')
    model_X = df_train

    # ---------------------------------------------> Import sklearn packages

    # Import preprocessing, selection and metrics
    from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectFromModel

    # Import classifiers
    from sklearn.dummy import DummyClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC

    # ---------------------------------------------> Split, transform, helper dataset

    def split_transform(*args, test_size=0.25, random_state=0, phase='train'):
        X, y = args
        not_class2 = (y != 2)
        scaler = StandardScaler() # RobustScaler()  # MinMaxScaler()
        if phase == 'train':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            # Multi class to binary
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

    # ---------------------------------------------> Final Model Fit

    def auc_scores(model, *args, k=5, threshold=0.50):
        """CV scores"""
        X, y = args
        predictions = cross_val_predict(model, X, y, cv=k, n_jobs=-1)
        pred_probas = (cross_val_predict(model, X, y, cv=k, method='predict_proba', n_jobs=-1)[:, 1] > threshold) * 1
        pred_logpro = (cross_val_predict(model, X, y, cv=k, method='predict_log_proba', n_jobs=-1)[:, 1] > threshold) * 1
        print('AUC - Test predict  {:.2%}'.format(roc_auc_score(y, predictions)))
        print('AUC - Test probabil {:.2%}'.format(roc_auc_score(y, pred_probas)))
        print('AUC - Test log prob {:.2%}'.format(roc_auc_score(y, pred_logpro)))
        return None

    # Training scores
    X_train, X_test, y_train, y_test = split_transform(model_X, model_y, test_size=0.1, phase='train')
    clf_train = GaussianNB().fit(X_train, y_train)
    pred_train = clf_train.predict(X_train)
    print('AUC - Train pred    {:.2%}'.format(roc_auc_score(y_train, pred_train)))

    # CV scores
    R = 43
    X_model, y_model = split_transform(model_X, model_y, phase='model')
    clf = GaussianNB().fit(X_model, y_model)
    auc_scores(clf, X_model, y_model, k=10, threshold=0.35)



    # clf1 = GradientBoostingClassifier(learning_rate=0.05, max_depth=2, random_state=0).fit(X_model, y_model)
    # auc_scores(clf1, X_model, y_model)

    # clf2 = RandomForestClassifier(random_state=R).fit(X_model, y_model)
    # auc_scores(clf2, X_model, y_model)

    # clf3 = DecisionTreeClassifier(random_state=R).fit(X_model, y_model)
    # auc_scores(clf3, X_model, y_model)

    # clf4 = Pipeline([
    #     ('feature_selection', SelectFromModel(LinearSVC(penalty='l1', loss='squared_hinge', dual=False, random_state=R))),
    #     ('classification', GaussianNB())
    # ])
    # clf4.fit(X_model, y_model)
    # auc_scores(clf4, X_model, y_model)

    # clf5 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    # auc_scores(clf5, X_model, y_model)

    # from sklearn.gaussian_process import GaussianProcessClassifier
    # from sklearn.gaussian_process.kernels import RBF
    # clf6 = GaussianProcessClassifier(1.0 * RBF(1.0))
    # auc_scores(clf6, X_model, y_model)

    # from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    # clf7 = AdaBoostClassifier()
    # auc_scores(clf7, X_model, y_model)

    # ---------------------------------------------> Test Final Model

    X_unseen = split_transform(df_test, None, test_size=0.2, random_state=R, phase='predict')

    predicted = clf.predict(X_unseen)
    pred_series = pd.DataFrame(predicted)
    pred_series['ticket_id'] = df_unseen['ticket_id']
    pred_series.set_index('ticket_id', inplace=True)

    pred_series.to_csv('detroit_blight2.csv')

    return None  # TODO pred_series

blight_model_()


# --------------------------------------------->  <-------------------------------------------- #



def blight_model():

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
    df_addresses = pd.read_csv('addresses.csv')
    df_latlons = pd.read_csv('latlons.csv')

    # ---------------------------------------------> Merge datasets on zip_code

    def merge_df(df_, df_latlon, df_address):
        df_zip = pd.merge(df_latlon, df_address, how='left', left_on='address', right_on='address')
        df_merge = pd.merge(df_, df_zip, how='left', left_on='ticket_id', right_on='ticket_id')
        return df_merge

    df_train = merge_df(df_train, df_latlons, df_addresses)
    df_test = merge_df(df_unseen, df_latlons, df_addresses)

    # ------------------------------------------ Feature Selection

    # Drop features
    garbage_features = ['ticket_id', 'violation_street_name', 'violation_street_number', 'violation_zip_code',
                        'violation_code', 'mailing_address_str_name', 'mailing_address_str_number',
                        'non_us_str_code', 'grafitti_status', 'state', 'city', 'address', 'agency_name',
                        'inspector_name', 'violator_name', 'country', 'zip_code', 'discount_amount', 'state_fee',
                        'clean_up_cost', 'discount_amount', 'admin_fee']
    df_test.drop(garbage_features, axis=1, inplace=True)
    df_train.drop(garbage_features, axis=1, inplace=True)

    train_only_features = ['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status',
                           'compliance_detail']
    df_train.drop(train_only_features, axis=1, inplace=True)

    # NaN imputation - train
    thres_train = 0.001
    nans_train = list(df_train.columns[df_train.isnull().mean() > thres_train])
    # assert nan_features == [], '{} need NaN imputation'.format(nan_features)
    df_train.dropna(axis=0, inplace=True)
    assert df_train.isnull().sum().sum() == 0, 'Not all NaNs are removed'

    # NaN imputation - test
    thres_test = 0
    nans_test = list(df_test.columns[df_test.isnull().mean() > thres_test])
    # assert nan_features == [], '{} need NaN imputation'.format(nan_features)
    for ftr in nans_test:
        df_test.loc[:, ftr] = df_test.loc[:, ftr].fillna(method='pad')
    assert df_test.isnull().sum().sum() == 0, 'Not all NaNs are removed'

    #
    df_train['compliance'] = df_train['compliance'].fillna(2).replace({'0.0': 0, '1.0': 1}).astype('category')
    M = (df_train.loc[:, 'compliance'] == 0) | (df_train.loc[:, 'compliance'] == 1)
    df_train = df_train.loc[M, :].copy()
    assert df_train.is_copy == None, 'Check if use proper copy'

    # ---------------------------------------------> EDA & Feature Engineering

    # List of categorical features to convert to dummy vars
    dummies = []

    # ---------------------------------------------> 'elapsed_days'
    ftr_h = 'hearing_date'
    ftr_t = 'ticket_issued_date'
    for df in [df_train, df_test]:
        df.drop(ftr_t, axis=1, inplace=True)
        df.drop(ftr_h, axis=1, inplace=True)

    # ---------------------------------------------> 'violation_code'
    # categorical
    ftr = 'violation_description'
    for df in [df_train, df_test]:
        # replace values of feature
        top_cat(df, ftr, 4)
    dummies.append(ftr)

    # ---------------------------------------------> 'fine_amount'
    # binary
    ftr = 'fine_amount'
    for df in [df_train, df_test]:
        # Replace values of feature
        top_cat(df, ftr, top=4)
    dummies.append(ftr)

    # ---------------------------------------------> 'judgment_amount'
    # binary
    ftr = 'judgment_amount'
    for df in [df_train, df_test]:
        # Replace values of feature
        top_cat(df, ftr, top=3)
    dummies.append(ftr)

    # ---------------------------------------------> 'late_fee'
    # binary
    ftr = 'late_fee'
    for df in [df_train, df_test]:
        thres_min, thres_max = -0.1, 1
        df['thres'] = (numerize_code(df, ftr, 0) > thres_max) | (numerize_code(df, ftr, 0) < thres_min)
        df[ftr] = df['thres'].astype('int')

    # ---------------------------------------------> 'disposition'
    # binary
    ftr = 'disposition'
    for df in [df_train, df_test]:
        # Replace values of feature
        top_cat(df, ftr, top=2)
    dummies.append(ftr)

    # ---------------------------------------------> Last Feature Cleaning

    for df in [df_train, df_test]:
        try:
            df.drop('thres', axis=1, inplace=True)
        except:
            continue

    df_train = pd.get_dummies(df_train, columns=dummies, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=dummies, drop_first=True)

    # ------------------------------------------ Final dataset

    # train_features = set(df.columns.tolist()) ^ set(['compliance'])
    model_y = df_train.pop('compliance')
    model_X = df_train

    # ---------------------------------------------> Import sklearn packages

    # Import preprocessing, selection and metrics
    from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler


    # Import classifiers
    from sklearn.naive_bayes import GaussianNB

    # ---------------------------------------------> Split, transform, helper dataset

    def split_transform(*args, test_size=0.25, random_state=0, phase='train'):
        X, y = args
        not_class2 = (y != 2)
        scaler = StandardScaler() # RobustScaler()  # MinMaxScaler()
        if phase == 'train':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            # Multi class to binary
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

    # ---------------------------------------------> Final Model Fit

    R = 43
    X_model, y_model = split_transform(model_X, model_y, phase='model')
    clf = GaussianNB().fit(X_model, y_model)

    # ---------------------------------------------> Test Final Model

    X_unseen = split_transform(df_test, None, test_size=0.2, random_state=R, phase='predict')

    predicted = clf.predict(X_unseen)
    pred_series = pd.DataFrame(predicted)
    pred_series['ticket_id'] = df_unseen['ticket_id']
    pred_series.set_index('ticket_id', inplace=True)

    return pred_series

blight_model()


# --------------------------------------------->  <-------------------------------------------- #
    # Fit to data and predict using pipelined scaling, GNB and PCA.
    # from sklearn.pipeline import make_pipeline
    # from sklearn.decomposition import PCA
    # X_train, X_test, y_train, y_test = split_transform(model_X, model_y, phase='train')
    # std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
    # std_clf.fit(X_train, y_train)
    # pred_test_std = std_clf.predict(X_test)
    #
    # # Show prediction accuracies in scaled and unscaled data.
    # print('\nPrediction accuracy for the standardized test dataset with PCA')
    # print('{:.2%}\n'.format(roc_auc_score(y_test, pred_test_std)))

    # --------------------------------------------->  <-------------------------------------------- #