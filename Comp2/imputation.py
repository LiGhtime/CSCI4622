# Deep learing imputation
# import datawig

import pandas as pd
import numpy as np

# read data
train_data = pd.read_csv("stock_XY_train.csv")
test_data = pd.read_csv("stock_X_test.csv")

train_data = train_data.drop(['id', 'Ticker', 'Sector'],axis=1).values
test_data = test_data.drop(['id', 'Unnamed: 0', 'Ticker', 'Sector'],axis=1).values

# train_data imputation
# df_train_tr, df_val_tr = datawig.utils.random_split(train_data)
# df_train_te, df_val_te = datawig.utils.random_split(test_data)

from impyute.imputation.cs import mice

# start the MICE training
imputed_training = mice(train_data)
print(train_data.shape)
print(imputed_training.shape)

imputed_testing = mice(test_data)
print(test_data.shape)
print(imputed_testing.shape)

# Output the results to files
imputed_training.to_csv("imputed_train.csv",index=False,sep=',')
imputed_testing.to_csv("imputed_test.csv",index=False,sep=',')
# f = open("imputed_train.csv", "w", newline = '') 
# writer = csv.writer(f)
# writer.writerow(())
# for i in range(len()):
#     writer.writerow(())
# f.close() 

# f = open("imputed_test.csv", "w", newline = '') 
# writer = csv.writer(f)
# writer.writerow(())
# for i in range(len()):
#     writer.writerow(())
# f.close() 


# Initialize a SimpleImputer model
# imputer = datawig.SimpleImputer(
#     input_columns=[], 
#     output_column= '0',
#     output_path = 'imputer_model'
#     )

# Imputer model on the train data
# imputer.fit(train_df=df_train_tr, num_epochs=5)

# Impute missing values and return original dataframe with predictions
# imputed_tr = imputer.predict(df_train_tr)
# print(imputed_tr.shape)
# Imputer model on the test data
# imputer.fit(train_df=df_train_te, num_epochs=5)

# Impute missing values and return original dataframe with predictions
# imputed_te = imputer.predict(df_val_te)
# print(imputed_te.shape)

# Imputation complete.


# https://github.com/awslabs/datawig/tree/master/examples
# https://datawig.readthedocs.io/en/latest/source/userguide.html#default-simpleimputer