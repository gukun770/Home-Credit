from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, Imputer, MinMaxScaler
import numpy as np
import pandas as pd
import sys

class PreselectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, preselect_cols):
        self.preselect_cols = preselect_cols

    def get_params(self, **kwargs):
        return {'preselect_cols': self.preselect_cols}

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.copy()
        # X = X.reset_index()
        X = X.loc[:,self.preselect_cols]
        return X

class CleanData(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        # Loan
        # X['AMT_CREDIT'] = np.log(X['AMT_CREDIT'])
        # X['AMT_ANNUITY'] = np.log(X['AMT_ANNUITY'])
        # X['AMT_GOODS_PRICE'] = np.log(X['AMT_GOODS_PRICE'])

        # Days --> Years
        X["DAYS_BIRTH"] = X["DAYS_BIRTH"]/-365
        # X.loc[X["DAYS_EMPLOYED"]>0,'DAYS_EMPLOYED']=1

        # DAYS_EMPLOYED
        X.loc[(X['NAME_INCOME_TYPE']=='Pensioner')&(
        X['DAYS_EMPLOYED']>0),'DAYS_EMPLOYED'] = -4000
        X.loc[X['NAME_INCOME_TYPE']=='Unemployed','DAYS_EMPLOYED'] = 0
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED']*-1

        # Phone
        X['DAYS_LAST_PHONE_CHANGE']=X['DAYS_LAST_PHONE_CHANGE']/-365

        # FLAG_OWN_CAR
        X['FLAG_OWN_CAR'] = X['FLAG_OWN_CAR'].replace({'N':0,'Y':1})

        # FLAG_OWN_REALTY
        # X['FLAG_OWN_REALTY'] = X['FLAG_OWN_REALTY'].replace({'N':0,'Y':1})

        # GENDER
        X['CODE_GENDER'] = X['CODE_GENDER'].replace({'F':0,'M':1, 'XNA':0})

        # OWN_CAR_AGE
        X['OWN_CAR_AGE'] = X['OWN_CAR_AGE'].fillna(7)

        return X

class Getdummies(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.X_train_transpose = None
        self.cat_columns = None
        self.non_cal_columns = None

    def fit(self, X, y):
        X_train = X.copy()
        self.cat_columns = X_train.loc[:,X_train.dtypes=='object'].columns.tolist()
        self.non_cal_columns = X_train.loc[:,~(X_train.dtypes=='object')].columns.tolist()
        X_train_cal_columns = pd.get_dummies(X_train[self.cat_columns],dummy_na=True)
        self.X_train_transpose = X_train_cal_columns.head(1).T
        self.X_train_transpose.columns =  [9999999]
        return self

    def transform(self, X):
        df = X.copy()
        df_cat = pd.get_dummies(df[self.cat_columns],dummy_na=True)
        df_non_cat = df[self.non_cal_columns]
        df_cat = self.X_train_transpose.join(df_cat.T, how='left').T
        df_cat = df_cat.fillna(0)
        df_cat = df_cat.iloc[1:,:]
        df = pd.concat([df_non_cat, df_cat], axis=1)
        return df

class BoolNan(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_columns_nan = None

    def fit(self, X, y):
        self.num_columns_nan = X.columns[(X.isna().any())&~(X.dtypes=='object')].tolist()
        return self

    def transform(self, X):
        df = X.copy()

        def add_nan(df, col_name):
            nan_series = df[col_name].isna()
            nan_series.name = df[col_name].name+'_nan'
            return pd.concat([df,nan_series], axis=1)

        for name in self.num_columns_nan:
            df = add_nan(df, name)
        return df

class ReplaceNaN(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        num_col_name = X.columns[~(X.dtypes=='object')].tolist()
        self.dict = X[num_col_name].mean().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        X = X.fillna(value=self.dict)
        checknan = X.columns[X.isna().sum()>0].tolist()
        if checknan:
            print(checknan)
            for col in checknan:
                X.drop(col, axis=1, inplace=True)
        return X

class FeatureEnginner(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        df_new = X.copy()

        # Repayment mutiples
        df_new['Fe:amt_credit_income_ratio'] = df_new['AMT_CREDIT']/df_new['AMT_INCOME_TOTAL']
        df_new['Fe:annuity_income_ratio'] = df_new['AMT_ANNUITY']/df_new["AMT_INCOME_TOTAL"]

        # Add annuity income ratio level
        df_new['Fe:mutiple_BU_credit_amt'] = df_new['BU_sum_AMT_CREDIT_SUM']/df_new['AMT_CREDIT']
        df_new['Fe:mutiple_BU_credit_amt_overdue'] = df_new['BU_sum_AMT_CREDIT_SUM_OVERDUE']/df_new['AMT_CREDIT']


        # Some interaction terms
        # df_new['Days_vs_source_1'] = df_new['EXT_SOURCE_1']*df_new['DAYS_BIRTH']
        # df_new['Days_vs_employed'] = df_new['DAYS_EMPLOYED']*df_new['DAYS_BIRTH']
        # df_new['EXT_SOURCE_2_vs_REGION_RATING_CLIENT_W_CITY'] = df_new['EXT_SOURCE_2']*df_new['REGION_RATING_CLIENT_W_CITY']
        return df_new

class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, list_to_drop):
        self.list_to_drop = list_to_drop

    def get_params(self, **kwargs):
        return {'list_to_drop': self.list_to_drop}

    def fit(self, X, y):
        return self

    def transform(self, X):
        df_new = X.copy()
        for name in self.list_to_drop:
            if name in df_new.columns:
                df_new.drop(name, axis=1,inplace=True)
        return df_new

class VsAverage(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mean_amt_annuity_per_occupation = None

    def fit(self, X, y):
        df = X.copy()

        self.mean_amt_annuity_per_occupation = df.groupby('OCCUPATION_TYPE')[['AMT_ANNUITY']].mean()
        return self

    def transform(self, X):
        df = X.copy()

        df = df.join(self.mean_amt_annuity_per_occupation, on='OCCUPATION_TYPE',
                    how='left', rsuffix='_per_occupation')
        df['Fe:diff_AMT_ANNUITY_per_occupation'] = df['AMT_ANNUITY'] - df['AMT_ANNUITY_per_occupation']
        df['Fe:diff_AMT_ANNUITY_per_occupation'] = df['diff_AMT_ANNUITY_per_occupation'].fillna(df['diff_AMT_ANNUITY_per_occupation'].mean())

        df.drop('AMT_ANNUITY_per_occupation', axis=1, inplace=True)
        return df

# if __name__ == '__main__':
#     application_train_original = pd.read_csv('data/application_train.csv.zip')
#     train =  application_train_original.copy()
#     application_test_original = pd.read_csv('data/application_test.csv.zip')
#     test = application_test_original.copy()
#     p = Pipeline([
#         ('filter', FilterColumns()),
#         ('clean', ColumnsTransformation())
#     ])     26441.027001 -24700.5