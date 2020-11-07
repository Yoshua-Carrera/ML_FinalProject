import pandas as pd
import numpy as np
import json
import ast

df = pd.read_csv('LOL/LeagueofLegends.csv')

print(df.info())

game_mins = []

for data in df['rDragons'][1:1000]:
    print(data)
