import json
import ast
import pprint

with open('championdata/Champion_role.json') as f:
    data = json.load(f)

for i in data:
    data[i] = ast.literal_eval(data[i])