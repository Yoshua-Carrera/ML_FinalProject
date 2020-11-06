import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('LOL/Clean_LeagueofLegends.csv')

for cols in df.columns:
    print('\nDatapoint #10 in {}: \n==>\t{}'.format(cols, df[cols][10]))