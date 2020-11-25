import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras import metrics

import pickle

""" 
Please run the data_cleaning script ONCE before running this one

X thanks X
"""

df = pd.read_csv('LOL/Clean_LeagueofLegends.csv')

x_cols = ['blueMiddleChamp', 'blueJungleChamp', 'redMiddleChamp', 'redJungleChamp',
          'rKills_pre15', 'rTowers_pre15', 'rDragons_pre15','rHeralds_pre15', 'golddiff_min15']

x_cols_v2 = ['blueJungleChampTags', 'redJungleChampTags', 'blueMiddleChampTags', 'redMiddleChampTags', 
            'blueADCChampTags', 'redADCChampTags', 'blueSupportChampTags', 'redSupportChampTags', 
            'blueTopChampTags', 'redTopChampTags', 'rKills_pre15', 'rTowers_pre15', 'rDragons_pre15',
            'rHeralds_pre15', 'golddiff_min15']

x = df[x_cols_v2]
y = df['rResult']

x = pd.get_dummies(x, columns=['blueJungleChampTags', 'redJungleChampTags', 'blueMiddleChampTags', 'redMiddleChampTags', 
                                'blueADCChampTags', 'redADCChampTags', 'blueSupportChampTags', 'redSupportChampTags', 
                                'blueTopChampTags', 'redTopChampTags'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

'''
Logistic model

Variables:
    ** Red_TOP: Red Champion selection for the TOP role (string) 
    Red_MID: Red Champion selection for the MID role (string) 
    ** Red_ADC: Red Champion selection for the ADC role (string) 
    ** Red_SUP: Red Champion selection for the SUP role (string) 
    Red_JUNG: Red Champion selection for the JUNG role (string) 
    ** Blue_TOP: Blue Champion selection for the TOP role (string) 
    Blue_MID: Blue Champion selection for the MID role (string) 
    ** Blue_ADC: Blue Champion selection for the ADC role (string) 
    ** Blue_SUP: Blue Champion selection for the SUP role (string) 
    Blue_JUNG: Blue Champion selection for the JUNG role (string)
    Red_Kills_pre15: number of kills before minute 15 for red (integer)
    Red_Towers_pre15: number of towers destroyed before minute 15 for red (integer)
    Red_dragons_pre15: number of dragons captured by minute 15 (integer)
    Red_herald_pre15: herald captured by minute 15 for red (integer)
    Gold_Diff_MIN_15: Diff in gold (red team - blue team) (integer)
Target:
    rResult: 1 if red team won (boolean)
'''

logit = LogisticRegression()
logit.fit(x_train, y_train)

y_pred = logit.predict(x_test)

logit_accuracy = sum(y_pred==y_test)/len(y_test)

print('Logistic model accuracy: {}'.format(logit_accuracy)) # 0.7355643044619422
print('='*50 + 'Confusion matrix' + '='*50, '\n', confusion_matrix(y_test, y_pred))

pickle.dump(logit, open('models/logit.pkl', 'wb'))


'''
Decision Tree

Variables:
    ** Red_TOP: Red Champion selection for the TOP role (string) 
    Red_MID: Red Champion selection for the MID role (string) 
    ** Red_ADC: Red Champion selection for the ADC role (string) 
    ** Red_SUP: Red Champion selection for the SUP role (string) 
    Red_JUNG: Red Champion selection for the JUNG role (string) 
    ** Blue_TOP: Blue Champion selection for the TOP role (string) 
    Blue_MID: Blue Champion selection for the MID role (string) 
    ** Blue_ADC: Blue Champion selection for the ADC role (string) 
    ** Blue_SUP: Blue Champion selection for the SUP role (string) 
    Blue_JUNG: Blue Champion selection for the JUNG role (string)
    Red_Kills_pre15: number of kills before minute 15 for red (integer)
    Red_Towers_pre15: number of towers destroyed before minute 15 for red (integer)
    Red_dragons_pre15: number of dragons captured by minute 15 (integer)
    Red_herald_pre15: herald captured by minute 15 for red (integer)
    Gold_Diff_MIN_15: Diff in gold (red team - blue team) (integer)
Target:
    rResult: 1 if red team won (boolean)
'''

dTree = DecisionTreeClassifier()

dTree.fit(x_train, y_train)

y_pred = dTree.predict(x_test)

dTree_accuracy = sum(y_pred==y_test)/len(y_test)

print('Decision tree model accuracy: {}'.format(dTree_accuracy)) # 0.6437007874015748
print('='*50 + 'Confusion matrix' + '='*50, '\n', confusion_matrix(y_test, y_pred))

pickle.dump(dTree, open('models/dTree.pkl', 'wb'))


'''
Random Forest

Variables:
    ** Red_TOP: Red Champion selection for the TOP role (string) 
    Red_MID: Red Champion selection for the MID role (string) 
    ** Red_ADC: Red Champion selection for the ADC role (string) 
    ** Red_SUP: Red Champion selection for the SUP role (string) 
    Red_JUNG: Red Champion selection for the JUNG role (string) 
    ** Blue_TOP: Blue Champion selection for the TOP role (string) 
    Blue_MID: Blue Champion selection for the MID role (string) 
    ** Blue_ADC: Blue Champion selection for the ADC role (string) 
    ** Blue_SUP: Blue Champion selection for the SUP role (string) 
    Blue_JUNG: Blue Champion selection for the JUNG role (string)
    Red_Kills_pre15: number of kills before minute 15 for red (integer)
    Red_Towers_pre15: number of towers destroyed before minute 15 for red (integer)
    Red_dragons_pre15: number of dragons captured by minute 15 (integer)
    Red_herald_pre15: herald captured by minute 15 for red (integer)
    Gold_Diff_MIN_15: Diff in gold (red team - blue team) (integer)
Target:
    rResult: 1 if red team won (boolean)
Number of trees:
    100
'''

rForest = RandomForestClassifier(n_estimators=100)

rForest.fit(x_train, y_train)

y_pred = rForest.predict(x_test)

rForest_accuracy = sum(y_pred==y_test)/len(y_test)

print('Random Forest model accuracy: {}'.format(rForest_accuracy)) # 0.7276902887139107
print('='*50 + 'Confusion matrix' + '='*50, '\n', confusion_matrix(y_test, y_pred))

pickle.dump(rForest, open('models/rForest.pkl', 'wb'))

'''
SVM

Variables:
    ** Red_TOP: Red Champion selection for the TOP role (string) 
    Red_MID: Red Champion selection for the MID role (string) 
    ** Red_ADC: Red Champion selection for the ADC role (string) 
    ** Red_SUP: Red Champion selection for the SUP role (string) 
    Red_JUNG: Red Champion selection for the JUNG role (string) 
    ** Blue_TOP: Blue Champion selection for the TOP role (string) 
    Blue_MID: Blue Champion selection for the MID role (string) 
    ** Blue_ADC: Blue Champion selection for the ADC role (string) 
    ** Blue_SUP: Blue Champion selection for the SUP role (string) 
    Blue_JUNG: Blue Champion selection for the JUNG role (string)
    Red_Kills_pre15: number of kills before minute 15 for red (integer)
    Red_Towers_pre15: number of towers destroyed before minute 15 for red (integer)
    Red_dragons_pre15: number of dragons captured by minute 15 (integer)
    Red_herald_pre15: herald captured by minute 15 for red (integer)
    Gold_Diff_MIN_15: Diff in gold (red team - blue team) (integer)
Target:
    rResult: 1 if red team won (boolean)
'''

svm = SVC(probability=True)

svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

print(x_test.head(1))

print(svm.predict_proba(x_test.head(1))[0])

svm_accuracy = sum(y_pred==y_test)/len(y_test)

print('SVM model accuracy: {}'.format(svm_accuracy)) # 0.7368766404199475
print('='*50 + 'Confusion matrix' + '='*50, '\n', confusion_matrix(y_test, y_pred))

pickle.dump(svm, open('models/svm.pkl', 'wb'))

'''
Neural Network

Variables:
    Red_MID: Red Champion selection for the MID role (string) 
    Red_JUNG: Red Champion selection for the JUNG role (string) 
    Blue_MID: Blue Champion selection for the MID role (string) 
    Blue_JUNG: Blue Champion selection for the JUNG role (string)
    Red_Kills_pre15: number of kills before minute 15 for red (integer)
    Red_Towers_pre15: number of towers destroyed before minute 15 for red (integer)
    Red_dragons_pre15: number of dragons captured by minute 15 (integer)
    Red_herald_pre15: herald captured by minute 15 for red (integer)
    Gold_Diff_MIN_15: Diff in gold (red team - blue team) (integer)
Target:
    rResult: 1 if red team won (boolean)
Architecture:
    * dense layer: 10 relu
    * dense layer: 10 relu
    * dense layer: 10 relu
    * dense layer: 5 relu
    * dense layer: 1 Sigmoid
'''

scalar = MinMaxScaler()
scalar.fit(x)

x_nn = scalar.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_nn, y, test_size=0.2, random_state=0)

nn = Sequential()
print('xtrain lenght: {}'.format(len(x_train[0])))
nn.add(Dense(10, input_dim=len(x_train[0]), activation='relu'))
nn.add(Dense(10, activation='relu'))
nn.add(Dense(10, activation='relu'))
nn.add(Dense(5, activation='relu'))
nn.add(Dense(1, activation='sigmoid'))

nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn.summary()

history = nn.fit(x_train, y_train, epochs=150, batch_size=10)

print(history.history.keys())

nn.evaluate(x_test, y_test) # 0.6640
y_pred = nn.predict(x_test)

nn.save('models/nn')

'''
Linear regression

Variables:
    ** Red_TOP: Red Champion selection for the TOP role (string) 
    Red_MID: Red Champion selection for the MID role (string) 
    ** Red_ADC: Red Champion selection for the ADC role (string) 
    ** Red_SUP: Red Champion selection for the SUP role (string) 
    Red_JUNG: Red Champion selection for the JUNG role (string) 
    ** Blue_TOP: Blue Champion selection for the TOP role (string) 
    Blue_MID: Blue Champion selection for the MID role (string) 
    ** Blue_ADC: Blue Champion selection for the ADC role (string) 
    ** Blue_SUP: Blue Champion selection for the SUP role (string) 
    Blue_JUNG: Blue Champion selection for the JUNG role (string)
    Red_Kills_pre15: number of kills before minute 15 for red (integer)
    Red_Towers_pre15: number of towers destroyed before minute 15 for red (integer)
    Red_dragons_pre15: number of dragons captured by minute 15 (integer)
    Red_herald_pre15: herald captured by minute 15 for red (integer)
    Gold_Diff_MIN_15: Diff in gold (red team - blue team) (integer)
Target:
    final_golddiff: Gold difference at the end of the game
'''

x = df[x_cols]
x = pd.get_dummies(x,columns=['blueMiddleChamp', 'blueJungleChamp', 'redMiddleChamp', 'redJungleChamp'])

y = df['golddiff_final']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

y_pred = linear_model.predict(x_test)

print('linear model R^2: {}'.format(linear_model.score(x_test, y_test))) # 0.4371161384443095

pickle.dump(linear_model, open('models/linear_model.pkl', 'wb'))
