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

from keras.models import Sequential
from keras.layers import Dense
from keras import metrics

df = pd.read_csv('LOL/Clean_LeagueofLegends.csv')

# for cols in df.columns:
#     print('\nDatapoint #100 in {}: \n==>\t{}'.format(cols, df[cols][100]))

print(df.columns)

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

x_cols = ['blueMiddleChamp', 'blueJungleChamp', 'redMiddleChamp', 'redJungleChamp',
          'rKills_pre15', 'rTowers_pre15', 'rDragons_pre15','rHeralds_pre15', 'golddiff_min15']

x = df[x_cols]
y = df['rResult']

x = pd.get_dummies(x,['blueMiddleChamp', 'blueJungleChamp', 'redMiddleChamp', 'redJungleChamp'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(x.head())
print(y.head())

logit = LogisticRegression()
logit.fit(x_train, y_train)

y_pred = logit.predict(x_test)

logit_accuracy = sum(y_pred==y_test)/len(y_test)

print('Logistic model accuracy: {}'.format(logit_accuracy)) # 0.7355643044619422

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
'''

rForest = RandomForestClassifier(n_estimators=100)

rForest.fit(x_train, y_train)

y_pred = rForest.predict(x_test)

rForest_accuracy = sum(y_pred==y_test)/len(y_test)

print('Decision tree model accuracy: {}'.format(rForest_accuracy)) # 0.7276902887139107

'''
Neural Network

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

scalar = MinMaxScaler()
scalar.fit(x)

x_nn = scalar.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_nn, y, test_size=0.2, random_state=0)

nn = Sequential()

nn.add(Dense(10, input_dim=len(x_train[0]), activation='relu'))
nn.add(Dense(50, activation='relu'))
nn.add(Dense(10, activation='relu'))
nn.add(Dense(5, activation='relu'))
nn.add(Dense(1, activation='sigmoid'))

nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn.summary()

history = nn.fit(x_train, y_train, epochs=500, batch_size=10) # 0.9826 (500 iterations)

print(history.history.keys())

nn.evaluate(x_test, y_test) 