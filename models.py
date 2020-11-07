import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('LOL/Clean_LeagueofLegends.csv')

# for cols in df.columns:
#     print('\nDatapoint #100 in {}: \n==>\t{}'.format(cols, df[cols][100]))

print(df.columns)

'''
Logistic model

Variables:
    Red_TOP: Red Champion selection for the TOP role (string) 
    Red_MID: Red Champion selection for the MID role (string) 
    Red_ADC: Red Champion selection for the ADC role (string) 
    Red_SUP: Red Champion selection for the SUP role (string) 
    Red_JUNG: Red Champion selection for the JUNG role (string) 
    Blue_TOP: Blue Champion selection for the TOP role (string) 
    Blue_MID: Blue Champion selection for the MID role (string) 
    Blue_ADC: Blue Champion selection for the ADC role (string) 
    Blue_SUP: Blue Champion selection for the SUP role (string) 
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

print(x.head())
print(y.head())

logit = LogisticRegression()
logit.fit(x, y)