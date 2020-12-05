from sklearn.impute import SimpleImputer
from pandas import read_csv
import numpy as np
import csv
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, Flatten, Activation
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten, Activation
from sklearn import preprocessing

# load dataset
dataframe = read_csv("stock_XY_train.csv")
dataframe = dataframe.drop(['id', 'Ticker', 'Sector'], axis=1)
dataset = dataframe.values

imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
dataset = imputer.fit_transform(dataset)
# dataset = preprocessing.normalize(dataset)

# train test spilt
labels = dataset[:, -1]
print("labels: ", labels)
features = dataset[:, :-1]
# print(features[0])
# print(features[1])
# print(features.shape)


def NNClassifier(features, labels):
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.33, random_state=1)
    # create model
    model = Sequential()
    model.add(Dense(222, input_dim=222, activation='relu'))
    # hidden layer
    model.add(Dropout(0.1))
    model.add(Dense(250, activation='relu'))
    # hidden layer
    # model.add(Dropout(0.3))
    # model.add(Dense(125, activation='relu'))
    # hidden layer
    # model.add(Dropout(0.2))
    # model.add(Dense(111, activation='relu'))
    # # hidden layer
    # model.add(Dropout(0.1))
    # model.add(Dense(64, activation='relu'))
    # # hidden layer
    # model.add(Dropout(0.1))
    # model.add(Dense(27, activation='relu'))
    # output layer
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=70, epochs=70,
                        verbose=2,
                        validation_data=(X_val, y_val))

    plt.figure(figsize=(20, 20))
    plt.plot(history.history['loss'], label='training data')
    plt.plot(history.history['val_loss'], label='validation data')
    plt.title('Data Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()

    return model


nn_classifer = NNClassifier(features, labels)

test_data = read_csv("stock_X_test.csv")
unnamed_col = test_data['Unnamed: 0']
test_data = test_data.drop(
    ['id', 'Unnamed: 0', 'Ticker', 'Sector'], axis=1).values
# test_data imputation
test_data = imputer.fit_transform(test_data)
# test_data = preprocessing.normalize(test_data)
# print(test_data)
# prediction
preds = nn_classifer.predict(test_data, batch_size=90, verbose=1)
preds = (preds > 0.5)
# print(uniq_column)
# print(preds.shape)
f = open("NNresult.csv", "w", newline='')
writer = csv.writer(f)
writer.writerow(('Unnamed: 0', 'Buy'))
for i in range(len(unnamed_col)):
    writer.writerow((unnamed_col[i], int(preds[i])))
f.close()

#And if your model suffers form dead neurons during training we should use leaky ReLu or Maxout function.
#https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f
#https://github.com/sesankmallikarjuna/neural-network-binary-classification/blob/master/cardiovascular_disease.ipynb
# #中间层普遍使用relu，输出层激活函数则是要根据任务配合loss来选择。
# 典型的多分类一般是softmax和交叉熵，二分类就是sigmoid和交叉熵
# 典型的回归用linear线性激活函数，loss用mse或者余弦距离