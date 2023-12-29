import pandas as pd
import matplotlib.pyplot as py
import matplotlib as mpl
import seaborn as sn
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np 

## Acquire Data
data = pd.read_csv('bankloans/bankloan.csv')
# data.info()

## Preprocessing
## Reindexing so that the label we are looking to predict Personal.Loan is the last column of our data set
data = data.reindex(columns=['ID', 'Age', 'Experience', 'Income', 'ZIP.Code', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities.Account', 'CD.Account', 'Online', 'CreditCard', 'Personal.Loan'])
# data.info()
## Separating into our label(y) and features(x)
y = data.iloc[:, -1]
x = data.iloc[:, 0:-1]

## dummy encoding our features
x = pd.get_dummies(x)

## train_test_split()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

## Standard Scaler - May want to use a column transformer
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


## Design Model
def design_model(num_features):
    model = Sequential()
    model.add(InputLayer(input_shape=(x_train.shape[1], )))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))

    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

## Call model on training data
cols = data.columns.tolist()
features, label = cols[:-1], cols[-1]

num_features = len(features)
model = design_model(num_features)

## Model Summary
print('Summary of Model:')
model.summary()
## 1,130 total/trainable params
## Early Stopping
es = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=10)
## Fit Model
model.fit(x_train, y_train, epochs=25, batch_size=30, verbose=1, validation_split=0.15, callbacks=[es])

## Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {loss}')
print(f'Test accuracy: {acc}')
# Classification Report
y_estimate = model.predict(x_test, verbose=0)
y_estimate = np.argmax(y_estimate, axis=1)

print(classification_report(y_test, y_estimate))

# Epoch 11: early stopping
# Test loss: 0.06001613289117813
# Test accuracy: 0.9826666712760925
#              precision    recall  f1-score   support
#
#           0       0.99      0.99      0.99      1351
#           1       0.90      0.93      0.91       149
#
#    accuracy                           0.98      1500
#   macro avg       0.95      0.96      0.95      1500
#weighted avg       0.98      0.98      0.98      1500 
# 
# 
# #