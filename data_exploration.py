import pandas as pd
import numpy as np
import json

df = pd.read_csv('LOL/LeagueofLegends.csv')

print(df.info())

game_mins = []

for cols in df.columns:
    print('\nDatapoint #10 in {}: \n==>\t{}'.format(cols, df[cols][10]))

for data in df['goldred']:
    game_mins.append(len(json.loads(data)))

print(sum(game_mins)/len(game_mins))