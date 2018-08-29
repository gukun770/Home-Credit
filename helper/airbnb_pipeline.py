import function as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import PredefinedSplit, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import re
import os
list_to_drop1 = []
list_to_drop2 = [
'property_type',
'zipcode',
'room_type',
 'property_type_House',
 'property_type_Apartment',
#  'bedrooms',
#  'room_type_Shared room',
 'property_type_Others',
 'guest_private_room',
#  'price_per_bedroom',
 ]
preselect_cols = [
        'price',
        'room_type',
        'guests_included',
        'property_type',
        'bedrooms',
        'cleaning_fee',
        'latitude',
        'longitude',
        'zipcode',
        ]
class PreselectColumns(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.reset_index()
        X = X.loc[:,preselect_cols]
        return X

class DataType(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        df = X.copy()
        # Clean price label: remove the dollar sign and comma.
        df.price=df.price.str.replace(r'[$,]','').astype(float)

        if 'cleaning_fee' in df.columns:
            df.cleaning_fee=df.cleaning_fee.str.replace(r'[$,]','').astype(float)

        # Convert 0 bed and 0 bedrooms to 1.
        if 'bedrooms' in df.columns:
            df.loc[df['bedrooms']==0,'bedrooms'] = 1
            df.bedrooms = df.bedrooms.fillna(1)

        # Convert minor cases for property type to 'other property types'
        if 'property_types' in df.columns:
            mask = df.property_type.isin(['Condominium','Condominium','Guest suite','Townhouse'])
            df.loc[mask,'property_type'] = 'Apartment'
            mask = df.property_type.isin(['Apartment','House'])
            df.loc[~mask, 'property_type'] = 'Others'

        # deal with nan
        if 'cleaning_fee' in df.columns:
            df.cleaning_fee = df.cleaning_fee.fillna(0)
        return df

class FeatureEnginner(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        df_new = X.copy()

        # # Compute price per person
        if 'guests_included' in df_new.columns:
            mask = df_new['guests_included'] == 0
            df_new.loc[mask, 'guests_included'] = 1
            df_new['price_per_person'] = df_new['price'] / df_new['guests_included']

        # Compute price per bedroom
        df_new['price_per_bedroom']=df_new['price']/df_new['bedrooms']

        # Cleaning fee per person
        df_new['cleaning_fee_person'] = df_new['cleaning_fee'] / df_new['guests_included']

        # Feature engineering - Compute distance from middle
        center = np.array([37.762835, -122.434239])
        if 'latitude' in df_new.columns and 'longitude' in df_new.columns:
            df_new['distance'] = df_new[['latitude', 'longitude']].apply(lambda row: np.linalg.norm(row - center), axis=1)

        return df_new

class Interaction(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self
    def transform(self, X):
        df = X.copy()
        df['price_entire_home'] = df['price'] * df['room_type_Entire home/apt']

        df['guest_entire_home'] = df['guests_included'] * df['room_type_Entire home/apt']
        df['guest_private_room'] = df['guests_included'] * df['room_type_Private room']
        df['cleanfee_entire_home'] = df['cleaning_fee'] * df['room_type_Entire home/apt']
        return df

class PricePerBedroom(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mean_train_df_without_month = None

    def fit(self, X, y):
        df = X.copy()

        self.mean_train_df_without_month = df.groupby('zipcode')[['price_per_bedroom']].mean()
        return self

    def transform(self, X):
        df = X.copy()

        df = df.join(self.mean_train_df_without_month, on='zipcode',
                    how='left', rsuffix='_per_neighbourhood')
        df['diff_price_per_bedroom_hood'] = df['price_per_bedroom'] - df['price_per_bedroom_per_neighbourhood']
        df['diff_price_per_bedroom_hood'] = df['diff_price_per_bedroom_hood'].fillna(df['diff_price_per_bedroom_hood'].mean())

        df.drop('price_per_bedroom_per_neighbourhood', axis=1, inplace=True)
        return df

class DropColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        df_new = X.copy()
        for name in list_to_drop1:
            if name in df_new.columns:
                df_new.drop(name, axis=1,inplace=True)
        return df_new

class Getdummies(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.X_train_transpose = None
        self.X_train = None
        self.X_test = None
        self.counter = 0

    def fit(self, X, y):
        X_train = X.copy()
        X_train = pd.get_dummies(X_train)
        for name in list_to_drop2:
            if name in X_train.columns:
                X_train.drop(name, axis=1,inplace=True)
        self.X_train_transpose = X_train.head(1).T
        self.X_train_transpose.columns =  [9999999]
        self.X_train = X_train.copy()
        return self

    def transform(self, X):
        df = X.copy()
        df = pd.get_dummies(df)
        df = self.X_train_transpose.join(df.T, how='left').T

        if df.isnull().sum().sum() > 0:
            nan_col = df.apply(lambda x:x.isnull().sum())
            print(nan_col[nan_col>0])
        df = df.fillna(0)
        df = df.iloc[1:,:]
        self.X_test = df.copy()
        return df