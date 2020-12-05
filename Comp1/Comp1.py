# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('./'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#import
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
import statistics as st
import csv

from scipy.stats import norm

from warnings import filterwarnings
filterwarnings('ignore')

train_data = pd.read_csv("train_data.csv")
val_data = pd.read_csv("val_data.csv")

#check all the features

# plt.scatter(train_data.price,train_data.sqft_living)
# plt.title("Price vs Square Feet")

# plt.figure(figsize=(10,10))
# sns.jointplot(x=train_data.lat.values, y=train_data.long.values, height=10)
# plt.ylabel('Longitude', fontsize=12)
# plt.xlabel('Latitude', fontsize=12)
# plt.show()
# sns.despine

# plt.scatter(train_data.price,train_data.long)
# plt.title("Price vs longitute")

# plt.scatter(train_data.price,train_data.lat)
# plt.xlabel("Price")
# plt.ylabel('Latitude')
# plt.title("Latitude vs Price")

# plt.scatter(train_data.bedrooms,train_data.price)
# plt.title("Bedroom and Price ")
# plt.xlabel("Bedrooms")
# plt.ylabel("Price")
# plt.show()
# sns.despine

# plt.scatter((train_data['sqft_living']+train_data['sqft_basement']),train_data['price'])

# plt.scatter(train_data.waterfront,train_data.price)
# plt.title("Waterfront vs Price ( 0= no waterfront)")

# train_data.floors.value_counts().plot(kind='bar')
# plt.scatter(train_data.floors,train_data.price)
# plt.scatter(train_data.condition,train_data.price)
# plt.scatter(train_data.zipcode,train_data.price)
# plt.title("Price VS zipcode")

# train_data.head()
# val_data.head()
# train_data.describe()

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

# Add model entries to model dictionary
models = {
            "Ridge": Ridge(),
            "Elastic Net": ElasticNet(),
            "XGBoost": xgb.XGBRegressor(objective ='reg:squarederror') 
         }

# , 
# min_child_weight = 5,
# gamma = 0.5,
# subsample = 1.0,
# colsample_bytree = 0.8,
# max_depth = 7,
# learning_rate = 0.1,
# scale_pos_weight = 0,
# reg_lambda = 5,
# n_estimators = 100

# K fold Validation
X, y = train_data.drop(['id', 'price'],axis=1).values, train_data.iloc[:,-1].values

# Normalization
# y = y.reshape(1, -1)
# y = preprocessing.normalize(y)

for ele in X:
    ele[1] = float(ele[1][0:8])
# print(X)
# print(y)
group = KFold(n_splits = 10, random_state = None, shuffle = True)

# training models and make predictions, calculate the average rmse, std and then print
for name, mod in zip(models.keys(), models.values()):
    rmse_list = []
    # print(mod.get_params())
    for train_index, test_index in group.split(X, y):
        # print(train_index.shape)
        # print(test_index.shape)
        # break
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = mod.fit(X_train, y_train)
        pred_y = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_y))
        rmse_list.append(rmse)
    avg = sum(rmse_list)/len(rmse_list)
    std = st.stdev(rmse_list)
    print(name, ": ", avg, "   ", std)        
     

# params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
# 'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4]}

# params = {
#         # 'min_child_weight': [1, 5, 10],
#         # 'gamma': [0.5, 1, 1.5, 2, 5],
#         # 'subsample': [0.6, 0.8, 1.0],
#         # 'colsample_bytree': [0.6, 0.8, 1.0],
#         # 'max_depth': [3, 4, 5]
#         'n_estimators': [100]
#         }

params = {
        'n_estimators':[10, 100, 1000, 5000],
        'min_child_weight': [1, 5, 10, 20, 50, 100],
        'gamma': [0, 0.1, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'learning_rate': [0.01, 0.07, 0.1, 0.5, 0.7, 1],
        'max_depth': [3:11],
        'colsample_bytree': [0.1, 0.3, 0.5, 0.6, 0.8, 1.0],
        'scale_pos_weight': [0, 0.01, 0.1, 0.5, 1, 2, 5],
        'reg_lambda': [0, 0.1, 0.5, 1, 2, 5, 10, 20],
        'eta': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
        }

# # Initialize XGB and GridSearch
# xgb = xgb.XGBRegressor(objective ='reg:squarederror', 
# min_child_weight = 5,
# gamma = 0.5,
# subsample = 1.0,
# colsample_bytree = 0.8,
# max_depth = 7,
# learning_rate = 0.1,
# scale_pos_weight = 0,
# reg_lambda = 5
# ) 

grid = GridSearchCV(xgb, params, scoring = 'neg_root_mean_squared_error', cv = 10, verbose=10)
grid.fit(X, y)
print("Best score: ", grid.best_score_)


# Initialize XGB and GridSearch
xgb = xgb.XGBRegressor(objective ='reg:squarederror', min_child_weight = 1, gamma = 0, 
                        learning_rate = 0.1, subsample = 0.8, max_depth = 7, 
                        colsample_bytree = 0.6, scale_pos_weight = 0, reg_lambda = 5) 

grid = GridSearchCV(xgb, params, scoring = 'neg_root_mean_squared_error', cv = 10, verbose=10)
grid.fit(X, y)
print("Best score: ", grid.best_score_)
print("\nBest n_estimators is", grid.best_estimator_.n_estimators)

# print("For XGB \nBest min_child_weight is:", grid.best_estimator_.min_child_weight, "\nBest gamma is", grid.best_estimator_.gamma, "\nBest subsample is", grid.best_estimator_.subsample, "\nBest colsample_bytree is", grid.best_estimator_.colsample_bytree, "\nBest max_depth is", grid.best_estimator_.max_depth)

# # XGB
# xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
#                           max_depth = 5, alpha = 10, n_estimators = 10)
# out_mod = xg_reg.fit(X, y)


val_X = val_data.drop(['id'],axis=1).values
for ele in val_X:
    ele[1] = float(ele[1][0:8])
# preds = xg_reg.predict(val_X)
preds = grid.best_estimator_.predict(val_X)
uniq_column = val_data['Unique_idx']
# print(uniq_column)

f = open("bestModel.csv", "w", newline = '') 
writer = csv.writer(f)
writer.writerow(('Unique_idx', 'price'))
for i in range(len(uniq_column)):
    writer.writerow((uniq_column[i], int(preds[i])))
f.close() 

# Light: 


# Liyang: https://www.kaggle.com/omarito/gridsearchcv-xgbregressor-0-556-lbn
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# Siyu: https://github.com/Shreyas3108/house-price-prediction/blob/master/housesales.ipynb
# 
# https://www.kaggle.com/swimmingwhale/xgboost-house-prices
# 