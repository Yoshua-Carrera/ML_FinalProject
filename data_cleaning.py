import pandas as pd
import numpy as np
import json
import ast
from pprint import pprint

df = pd.read_csv('LOL/LeagueofLegends.csv')

df_champs =  pd.read_csv('LOL/riot_champion.csv')

Champ_cols = ['blueJungleChamp', 'redJungleChamp', 'blueMiddleChamp', 'redMiddleChamp', 'blueADCChamp', 
            'redADCChamp', 'blueSupportChamp', 'redSupportChamp', 'blueTopChamp', 'redTopChamp']

New_Champ_cols = ['blueJungleChampTags', 'redJungleChampTags', 'blueMiddleChampTags', 'redMiddleChampTags', 'blueADCChampTags', 
                'redADCChampTags', 'blueSupportChampTags', 'redSupportChampTags', 'blueTopChampTags', 'redTopChampTags']

for role, roletag in zip(Champ_cols, New_Champ_cols):
    df[roletag] = pd.merge(
        df,
        df_champs[['id', 'tags']], 
        how='left', 
        left_on=role, 
        right_on = 'id'
    )['tags']  

print(df.info())

for cols in df.columns:
    print('\nDatapoint #10 in {}: \n==>\t{}'.format(cols, df[cols][10]))

print('\n' + '='*100)
print('='*100, '\n')

count_cols = ['goldred', 'bKills', 'rKills', 'bTowers', 'rTowers', 'bInhibs', 'rInhibs', 
            'bDragons', 'rDragons', 'bBarons', 'rBarons', 'bHeralds', 'rHeralds']

champ_gold_cols = ['goldblueTop', 'goldblueMiddle', 'goldblueJungle', 'goldblueSupport', 'goldblueADC', 
                    'goldredTop', 'goldredMiddle', 'goldredJungle', 'goldredSupport', 'goldredADC']

# goldred - goldblue

for cols in count_cols:
    if cols in ['goldred', 'goldblue']:
        df[cols] = df.apply(lambda row: json.loads(row[cols]), axis=1)
    else:
        df[cols] = df.apply(lambda row: ast.literal_eval(row[cols]), axis=1)
        df[str(cols+'_pre15')] = df.apply(lambda row: sum([1 if int(i[0]) <= 15 else 0 for i in row[cols]]), axis=1)


for cols in champ_gold_cols:
    df[cols] = df.apply(lambda row: json.loads(row[cols]), axis=1)

players = ['TopgoldDiff', 'MidgoldDiff', 'JunglegoldDiff', 'SupportgoldDiff', 'ADCgoldDiff']
blue_players_gold = ['goldblueTop', 'goldblueMiddle', 'goldblueJungle', 'goldblueSupport', 'goldblueADC']
red_players_gold = ['goldredTop', 'goldredMiddle', 'goldredJungle', 'goldredSupport', 'goldredADC']

for rChampG, bChampG, Champ in zip(red_players_gold, blue_players_gold, players):
    df[Champ] = df.apply(lambda row: np.subtract(row[rChampG], row[bChampG]), axis=1)
    pass

df['golddiff_min15'] = df.apply(lambda row: json.loads(row['golddiff'])[15], axis=1)
df['golddiff_final'] = df.apply(lambda row: json.loads(row['golddiff'])[-1], axis=1)

# Data join #

for cols in df.columns:
    print('\nDatapoint #10 in {}: \n==>\t{}'.format(cols, df[cols][10]))

df.to_csv('LOL/Clean_LeagueofLegends.csv')

# champion_tags = {}

# for row in range(len(df_champs)):
#     champion_tags[df_champs['id'][row]] = champion_tags[df_champs['tags'][row]]


# with open('championdata/Champion_tag.json', 'w') as fp:
#     json.dump(champion_tags, fp)

champion_tags = {}

for row in range(len(df_champs)):
    champion_tags[df_champs['id'][row]] = df_champs['tags'][row]

with open('championdata/Champion_tag.json', 'w') as fp:
    json.dump(champion_tags, fp, indent=4)

champion_roles = {}

for role in ['Jungle', 'Middle', 'ADC', 'Top', 'Support']:
    champion_roles[role] = str(set(
        pd.concat(
            [
                df['blue' + role + 'Champ'], 
                df['red' + role + 'Champ']
            ]
        )
    )).replace('{', '[').replace('}', ']')

with open('championdata/Champion_role.json', 'w') as fp:
    json.dump(champion_roles, fp, indent=4)