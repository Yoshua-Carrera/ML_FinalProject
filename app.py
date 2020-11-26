# This is the app
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import ast
import json

app = Flask(__name__)
model = pickle.load(open('models/svm.pkl', 'rb'))
df = pd.read_csv('LOL/Clean_LeagueofLegends.csv')

with open('championdata/Champion_role.json') as f:
    role_dict = json.load(f)

with open('championdata/Champion_tag.json') as f:
    tag_dict = json.load(f)

for role in role_dict:
    role_dict[role] = ast.literal_eval(role_dict[role])

x_cols = ['blueMiddleChamp', 'blueJungleChamp', 'redMiddleChamp', 'redJungleChamp',
        'rKills_pre15', 'rTowers_pre15', 'rDragons_pre15','rHeralds_pre15', 'golddiff_min15']
x = df[x_cols]
x = pd.get_dummies(x, columns=['blueMiddleChamp', 'blueJungleChamp', 'redMiddleChamp', 'redJungleChamp'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    ['rKills_pre15', 'rTowers_pre15', 'rDragons_pre15', 'rHeralds_pre15',
       'golddiff_min15', 'blueMiddleChamp_Ahri', 'blueMiddleChamp_Akali',  
       'blueMiddleChamp_Anivia', 'blueMiddleChamp_AurelionSol',
       'blueMiddleChamp_Azir',
       ...
       'redJungleChamp_Shyvana', 'redJungleChamp_Sion',
       'redJungleChamp_Skarner', 'redJungleChamp_Trundle',
       'redJungleChamp_Udyr', 'redJungleChamp_Vi', 'redJungleChamp_Volibear',
       'redJungleChamp_Warwick', 'redJungleChamp_XinZhao',
       'redJungleChamp_Zac']
    '''
    user_response = request.form.values()
    
    for i in range(5, len(user_response)+1):
        user_response[i] = role_dict[user_response[i]]
    
    features = [y+x.capitalize() if y != '' else int(x) for x, y in zip(request.form.values(), ['', '', '', '', '', 
                                                                                                'redTopChamp_', 'redMiddleChamp_', 'redJungleChamp_', 'redADCChamp_', 'redSupportChamp_', 
                                                                                                 'blueTopChamp_', 'blueMiddleChamp_', 'blueJungleChamp_', 'blueADCChamp_', 'blueSupportChamp_'])]
    input_array = [0]*len(x.columns)

    for i in range(len(x.columns)):
        if pd.Series.between(i, 0, 4):
            input_array[i] = features[i]
        elif x.columns[i] in features:
            print('champ {} in {}'.format(x.columns[i], i))
            input_array[i] = 1    
        else:
            input_array[i] = 0

    prediction = model.predict_proba([np.array(input_array)])
    print([np.array(input_array)])
    print(prediction)
    print(model.predict([np.array(input_array)])[0])
    output = round(prediction[0][1], 3)

    return render_template('index.html', prediction_text='Your probability of winning is: {:.4%} for the input: {}'.format(output, features))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    :return:
    '''

if __name__ == "__main__":
    app.run(debug=True)

