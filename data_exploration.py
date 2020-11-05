import pandas as pd
import numpy as np
import json
import ast

df = pd.read_csv('LOL/LeagueofLegends.csv')

print(df.info())

game_mins = []

for cols in df.columns:
    print('\nDatapoint #10 in {}: \n==>\t{}'.format(cols, df[cols][10]))

# for data in df['goldred']:
#     game_mins.append(len(json.loads(data)))

# print(sum(game_mins)/len(game_mins))

# df['goldred'] = df['goldred'].apply(lambda row: len(row), axis=1)

print('\n' + '='*100)
print('='*100, '\n')

count_cols = ['goldred', 'bKills', 'rKills', 'bTowers', 'rTowers', 'bInhibs', 'rInhibs', 
            'bDragons', 'rDragons', 'bBarons', 'rBarons', 'bHeralds', 'rHeralds']

champ_gold_cols = ['goldblueTop', 'goldblueMiddle', 'goldblueJungle', 'goldblueSupport', 'goldblueADC', 
                    'goldredTop', 'goldredMiddle', 'goldredJungle', 'goldredSupport', 'goldredADC']

# goldred - goldblue

for cols in count_cols:
    try:
        df[cols] = df.apply(lambda row: len(json.loads(row[cols])), axis=1)
    except:
        df[cols] = df.apply(lambda row: len(ast.literal_eval(row[cols])), axis=1)

for cols in champ_gold_cols:
    df[cols] = df.apply(lambda row: json.loads(row[cols]), axis=1)

players = ['TopgoldDiff', 'MidgoldDiff', 'JunglegoldDiff', 'SupportgoldDiff', 'ADCgoldDiff']
blue_players_gold = ['goldblueTop', 'goldblueMiddle', 'goldblueJungle', 'goldblueSupport', 'goldblueADC']
red_players_gold = ['goldredTop', 'goldredMiddle', 'goldredJungle', 'goldredSupport', 'goldredADC']

for rChampG, bChampG, Champ in zip(red_players_gold, blue_players_gold, players):
    df[Champ] = df.apply(lambda row: np.subtract(row[rChampG], row[bChampG]), axis=1)
    pass


for cols in df.columns:
    print('\nDatapoint #10 in {}: \n==>\t{}'.format(cols, df[cols][10]))

df.to_csv('LOL/Clean_LeagueofLegends.csv')