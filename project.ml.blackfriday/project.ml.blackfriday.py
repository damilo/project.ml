# Contest: Black Friday - Analytics Vidhya
# https://datahack.analyticsvidhya.com/contest/black-friday

# Daniel Hellwig / daniel.hellwig.p@gmail.com

# Solution written in Python v3.6.4
# package versions:
# numpy         1.15.2
# pandas        0.23.4
# scikit-learn  0.20.0


# imports
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings ("ignore", category=FutureWarning)


DATA_TRAIN = os.path.join ('data', 'train.csv')
DATA_TEST = os.path.join ('data', 'test.csv')
FILE_SUBMISSION = os.path.join ('data', 'project.ml.blackfriday.submission.csv')


# function defs
def rmse_metric (y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean ()
    return np.sqrt (mse)


# Data Exploration
# not submitted


# Data Preprocessing
print ('Step:', 'Data Preprocessing', '...', end = ' ', flush = True)

data_raw = pd.read_csv (DATA_TRAIN)

# replace NaN values with 0
fillna_cols = ['Product_Category_2', 'Product_Category_3']
data_raw[fillna_cols] = data_raw[fillna_cols].fillna (0)

# convert to same type as Product_Category_1
data_raw[fillna_cols] = data_raw[fillna_cols].astype (data_raw['Product_Category_1'].dtype)

# characteristics of data scales
data_char = defaultdict (list)
for col in data_raw.keys ():
    data_char[col] = np.sort (data_raw[col].unique ())

# append new category to existent data_char
data_char['Product_Category_1'] = np.append (0, data_char['Product_Category_1'])

# convert to type category
cat_cols = ['Age', 'Product_ID', 'Occupation', 'Gender', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']
for col in data_raw[cat_cols].keys ():
    data_raw[col] = data_raw[col].astype ('category', categories=data_char[col])

# 'Product_Category_2', 'Product_Category_3' are same as Product_Category_1
prod_cols = ['Product_Category_1', 'Product_Category_2', 'Product_Category_3']
data_raw[prod_cols] = data_raw[prod_cols].astype ('category', categories=data_char['Product_Category_1'])
# set categories of Product_Category_1
data_raw['Product_Category_1'] = data_raw['Product_Category_1'].cat.set_categories (data_char['Product_Category_1'])
data_raw['Product_Category_2'] = data_raw['Product_Category_2'].cat.set_categories (data_char['Product_Category_1'])
data_raw['Product_Category_3'] = data_raw['Product_Category_3'].cat.set_categories (data_char['Product_Category_1'])

# make a copy - not needed, just for my information copied from Jupyter Notebook
data_wrk = data_raw.copy (deep = True)

for col in data_wrk.columns:
    if (str (data_wrk[col].dtype) == 'category'):
        data_wrk[col] = data_wrk[col].values.codes

# Make X and y for training
drop_cols = ['User_ID', 'Purchase']
X = data_wrk.drop (drop_cols, axis=1).values
y = data_wrk['Purchase'].values

print ('done')


# Implementation
print ('Step:', 'Implementation', '...', end = ' ', flush = True)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# shuffle data and split into train and test set
X_train, X_val, y_train, y_val = train_test_split (
    X,
    y,
    test_size = 0.25,
    shuffle = True,
    random_state = 42
)

# feed data into learning algorithm
# remark: best estimator was found beforehand via grid search + cross-validation
dtr = DecisionTreeRegressor (
    criterion = 'mse',
    max_depth = 12
)

dtr.fit (X_train, y_train)

print ('done')


# Evaluation
print ('Step:', 'Evaluation', '...')

print ('  R2 score', 'train', dtr.score (X_train, y_train))
print ('  R2 score', 'val  ', dtr.score (X_val, y_val))

y_train_pred = dtr.predict (X_train)
y_val_pred = dtr.predict (X_val)
print ('  RMSE', 'train', rmse_metric (y_train, y_train_pred))
print ('  RMSE', 'val', rmse_metric (y_val, y_val_pred))

print ('done')


# Prediction
print ('Step:', 'Prediction', '...')

print ('  Data Preprocessing', '...', end = ' ', flush = True)
test_data_raw = pd.read_csv (DATA_TEST)

# copy data to submit
subm_cols = ['User_ID', 'Product_ID']
data_subm = test_data_raw[subm_cols]

# replace NaN values with 0
fillna_cols = ['Product_Category_2', 'Product_Category_3']
test_data_raw[fillna_cols] = test_data_raw[fillna_cols].fillna (0)

# convert to same type as Product_Category_1
test_data_raw[fillna_cols] = test_data_raw[fillna_cols].astype (test_data_raw['Product_Category_1'].dtype)

# convert to type category
cat_cols = ['Age', 'Product_ID', 'Occupation', 'Gender', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']
for col in test_data_raw[cat_cols].keys ():
    test_data_raw[col] = test_data_raw[col].astype ('category', categories=data_char[col])

# 'Product_Category_2', 'Product_Category_3' are same as Product_Category_1
prod_cols = ['Product_Category_1', 'Product_Category_2', 'Product_Category_3']
test_data_raw[prod_cols] = test_data_raw[prod_cols].astype ('category', categories=data_char['Product_Category_1'])
# set categories of Product_Category_1
test_data_raw['Product_Category_1'] = test_data_raw['Product_Category_1'].cat.set_categories (data_char['Product_Category_1'])
test_data_raw['Product_Category_2'] = test_data_raw['Product_Category_2'].cat.set_categories (data_char['Product_Category_1'])
test_data_raw['Product_Category_3'] = test_data_raw['Product_Category_3'].cat.set_categories (data_char['Product_Category_1'])

for col in test_data_raw.columns:
    if (str (test_data_raw[col].dtype) == 'category'):
        test_data_raw[col] = test_data_raw[col].values.codes

# Make X and y for training
drop_cols = ['User_ID']
X_test = test_data_raw.drop (drop_cols, axis=1).values

print ('done')

print ('  Predicting \'Purchase\'', '...', end = ' ', flush = True)

y_test = dtr.predict (X_test)

print ('done')

print ('  Saving results to submission file', '...', end = ' ', flush = True)

data_subm['Purchase'] = pd.Series (y_test, index = data_subm.index)

data_subm.to_csv (FILE_SUBMISSION, index = False)

print ('done')


print ('done') # from Step: Prediction


print ('This is the end ;), BR Daniel')