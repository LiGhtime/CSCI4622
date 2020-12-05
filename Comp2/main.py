from keras.models import Model
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.models import Sequential
from keras.datasets import mnist
import keras
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import statistics as st
from math import sqrt

# Dealing with missing data.
train_data = pd.read_csv("stock_XY_train.csv")
val_data = pd.read_csv("stock_X_test.csv")

train_data = train_data.drop(['id', 'Ticker', 'Sector'], axis=1).values
val_data = val_data.drop(['id', 'Unnamed: 0', 'Ticker', 'Sector'], axis=1).values

print("Start Impution")
imputer = SimpleImputer()
train_data = imputer.fit_transform(train_data)
val_data = imputer.fit_transform(val_data)


models = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier()
}

# model_NN = Classifier() # Neural Networks
X, y = train_data[:, :-1], train_data[:, -1]

# f = open("test.csv", "w", newline = '')
# writer = csv.writer(f)
# for i in range(X.shape[0]):
#    writer.writerow((X[i], y[i]))
# print(X)
# print(y)
# group = KFold(n_splits=2, random_state=None, shuffle=True)

# training models and make predictions, calculate the average rmse, std and then print
# for name, mod in zip(models.keys(), models.values()):
#     loss_list = []
#     # print(mod.get_params())
#     print("Start ", name)
#     for train_index, test_index in group.split(X, y):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         model = mod.fit(X_train, y_train)
#         pred_y = model.predict(X_test)
#         loss = log_loss(y_test, pred_y)
#         loss_list.append(loss)
#     avg = sum(loss_list)/len(loss_list)
#     std = st.stdev(loss_list)
#     print(name, ": ", avg, "   ", std)

params = {
		# 'n_estimators': [10, 100],
		# 'max_depth': [100, 1000],
        # 'min_samples_split': [2, 10],
        # 'min_samples_leaf': [1, 5, 10]
		'n_estimators': [1200],
		# 'max_feature' : [15],
		'max_depth': [1000],
        'min_samples_split': [2],
        'min_samples_leaf': [5]
        }

r_f = RandomForestClassifier()

grid = GridSearchCV(r_f, params, scoring = 'neg_log_loss', cv = 10, verbose=10)
grid.fit(X, y)
print("Best score: ", grid.best_score_)
print("\nBest n_estimators is", grid.best_estimator_.n_estimators)
print("\nBest max_depth is", grid.best_estimator_.max_depth)
print("\nBest min_samples_split is", grid.best_estimator_.min_samples_split)
print("\nBest min_samples_leaf is", grid.best_estimator_.min_samples_leaf)

preds = grid.best_estimator_.predict(val_data)
comp_data = pd.read_csv("stock_X_test.csv")
uniq_column = comp_data['Unnamed: 0']

# Output the results.
f = open("bestResult_RF.csv", "w", newline = '') 
writer = csv.writer(f)
writer.writerow(('Unnamed: 0', 'Buy'))
for i in range(len(uniq_column)):
    writer.writerow((uniq_column[i], int(preds[i])))
f.close() 

# https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
# https://github.com/rahulguptagzb09/ANN-Keras-Binary-Classification/blob/master/ann.py
# rf_random.best_params_
# https://scikit-learn.org/stable/modules/model_evaluation.html
#（1）二分类评判指标：f1，roc_auc （2）多分类评判指标：f1_weighted