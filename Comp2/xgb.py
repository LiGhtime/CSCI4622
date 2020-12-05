import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import statistics as st
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv("stock_XY_train.csv")
val_data = pd.read_csv("stock_X_test.csv")

train_data = train_data.drop(['id', 'Ticker', 'Sector'],axis=1).values
val_data = val_data.drop(['id', 'Unnamed: 0', 'Ticker', 'Sector'],axis=1).values

X, y = train_data[:, :-1], train_data[:,-1]
# print(X)
# print(y)
group = KFold(n_splits = 10, random_state = None, shuffle = True)

# GridSearch
params = {
        # 'n_estimators':[10, 100],
        # 'min_child_weight': [1, 5, 10],
        # 'gamma': [0, 1, 5],
        # 'learning_rate': [0.1, 0.5, 1],
        # 'max_depth': [3, 7, 10],
        # 'colsample_bytree': [0.1, 0.5, 1.0],
        # 'scale_pos_weight': [0, 1, 5],
        # 'reg_lambda': [0, 1, 5, 20],
        # 'eta': [0.01, 0.3, 0.5]
        'n_estimators': [100],
        'max_depth': [7],
        'learning_rate': [0.1],
        'min_child_weight': [10],
        'gamma': [5.9]
        }

xgb = XGBClassifier()
grid = GridSearchCV(xgb, params, scoring = 'neg_log_loss', cv = 10, verbose=10)
grid.fit(X, y)
print("Best score: ", grid.best_score_)
print("\nBest n_estimators is", grid.best_estimator_.n_estimators)
print("\nBest max_depth is", grid.best_estimator_.max_depth)
print("\nBest learning_rate is", grid.best_estimator_.learning_rate)
print("\nBest min_child_weight is", grid.best_estimator_.min_child_weight)
print("\nBest gamma is", grid.best_estimator_.gamma)

# training models and make predictions, calculate the average rmse, std and then print
# name = 'xgboost'
# mod = XGBClassifier()
# loss_list = []
# # print(mod.get_params())
# for train_index, test_index in group.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     model = mod.fit(X_train, y_train, verbose=1)
#     pred_y = model.predict(X_test)
#     loss = log_loss(y_test, pred_y)
#     loss_list.append(loss)
# avg = sum(loss_list)/len(loss_list)
# std = st.stdev(loss_list)
# print(name, ": ", avg, "   ", std)   

#preds = mod.predict(val_data)
xgb_dudoo = XGBClassifier()
preds = xgb_dudoo.predict(val_data)
preds = grid.best_estimator_.predict(val_data)
comp_data = pd.read_csv("stock_X_test.csv")
uniq_column = comp_data['Unnamed: 0']

# Output the results.
f = open("bestModel_xgb.csv", "w", newline = '') 
writer = csv.writer(f)
writer.writerow(('Unnamed: 0', 'Buy'))
for i in range(len(uniq_column)):
    writer.writerow((uniq_column[i], int(preds[i])))
f.close() 