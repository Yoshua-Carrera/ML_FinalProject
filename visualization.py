from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json

df = pd.read_csv('LOL/Clean_LeagueofLegends.csv')

# print(df.columns)

df['golddiff'] = df.apply(lambda row: json.loads(row['golddiff']), axis=1)

print(df[(df['bResult']==1) & (df['golddiff'].map(len) == 30)]["golddiff"])

print(np.average(df[(df['bResult']==1) & (df['golddiff'].map(len) == 30)]["golddiff"].values.tolist(), axis=0))
print(np.average(df[(df['bResult']==0) & (df['golddiff'].map(len) == 30)]["golddiff"].values.tolist(), axis=0))

fog, ax = plt.subplots()

ax.plot(
    np.average(df[(df['bResult']==1) & (df['golddiff'].map(len) == 30)]["golddiff"].values.tolist(), axis=0),
    label='Gold difference of winning team'
    )
ax.plot(
    np.average(df[(df['bResult']==0) & (df['golddiff'].map(len) == 30)]["golddiff"].values.tolist(), axis=0),
    label='Gold difference of losing team'
    )
legend = ax.legend(loc='upper left', shadow=True)

plt.xlabel('Minute')
plt.title('Gold Difference by the minute')

plt.show()

