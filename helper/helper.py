import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option("display.max_colwidth",999)
pd.set_option("display.max_rows",999)
pd.set_option("display.max_columns",999)

def show_corr(col_name,df_corr, df, nb=6):

    if df.dtypes[col_name] == 'object':
        names = df[col_name].unique()
        names = ['{}_{}'.format(col_name, i) for i in names if i is not np.nan]
        for name in names:
            corr_table = pd.DataFrame({'Most positive corr column':df_corr[name].sort_values(ascending=False)[:nb].index,
                                   'Most positive corr value':df_corr[name].sort_values(ascending=False)[:nb].values,
                                   'Most negative corr column': df_corr[name].sort_values(ascending=True)[:nb].index,
                                   'Most negative corr value': df_corr[name].sort_values(ascending=True)[:nb].values})
            print(name)
            print(corr_table)
            print('\n')

    else:
        corr_table = pd.DataFrame({'Most positive corr column':df_corr[col_name].sort_values(ascending=False)[:nb].index,
                                   'Most positive corr value':df_corr[col_name].sort_values(ascending=False)[:nb].values,
                                   'Most negative corr column': df_corr[col_name].sort_values(ascending=True)[:nb].index,
                                   'Most negative corr value': df_corr[col_name].sort_values(ascending=True)[:nb].values})
        return corr_table

def report_nan(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    df_missing  = pd.concat([total, percent], axis=1,
                                                keys=['Total', 'Percent'])
    return df_missing

def report_categories(df):
    mask = df.dtypes == object
    return df[df.dtypes[mask].index].describe()

def report_numeric(df):
    mask = df.dtypes == object
    return df[df.dtypes[~mask].index].describe()

def plot_data(col_name, df):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    sns.boxplot(df[col_name], ax=ax1)
    sns.distplot(df[col_name].dropna(), ax=ax2)
    plt.show();

def check_subcat_cols(df_train, df_test, nb=20):
    object_cols = df_train.dtypes[df_train.dtypes == object].index.values
    object_cols = [col for col in object_cols if len(df_train[col].unique())<=nb]
    container={}
    d = {}
    for col in object_cols:
        d[col] = set(df_train[col].unique())
        container['train'] = d
    d = {}
    for col in object_cols:
        d[col]= set(df_test[col].unique())
        container['test']=d
    for col in object_cols:
        train_test = (container['train'][col] - container['test'][col])
        if train_test:
            print('Classes in train but not in test: {} : {} \n'.format(col, list(train_test)))
        test_train = (container['test'][col] - container['train'][col])
        if test_train:
            print('Classes in test but not in train: feature {} : {} \n'.format(col, list(test_train)))
    return object_cols