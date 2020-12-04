# This is the app
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import ast
import json
from flask_wtf import FlaskForm
from wtforms import SelectField


app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secret'
model = pickle.load(open('models/svm.pkl', 'rb'))
df = pd.read_csv('LOL/Clean_LeagueofLegends.csv')

with open('championdata/Champion_role.json') as f:
    role_dict = json.load(f)

with open('championdata/Champion_tag.json') as f:
    tag_dict = json.load(f)

for role in role_dict:
    role_dict[role] = ast.literal_eval(role_dict[role])

class Form(FlaskForm):
    redTopChamp = SelectField('redTopChamp', choices=[(champ, champ) for champ in role_dict['Top']])
    redJungleChamp = SelectField('redJungleChamp', choices=[(champ, champ) for champ in role_dict['Jungle']])
    redMiddleChamp = SelectField('redMiddleChamp', choices=[(champ, champ) for champ in role_dict['Middle']])
    redADCChamp = SelectField('redADCChamp', choices=[(champ, champ) for champ in role_dict['ADC']])
    redSupportChamp = SelectField('redSupportChamp', choices=[(champ, champ) for champ in role_dict['Support']])
    blueTopChamp = SelectField('redTopChamp', choices=[(champ, champ) for champ in role_dict['Top']])
    blueJungleChamp = SelectField('redJungleChamp', choices=[(champ, champ) for champ in role_dict['Jungle']])
    blueMiddleChamp = SelectField('redMiddleChamp', choices=[(champ, champ) for champ in role_dict['Middle']])
    blueADCChamp = SelectField('redADCChamp', choices=[(champ, champ) for champ in role_dict['ADC']])
    blueSupportChamp = SelectField('redSupportChamp', choices=[(champ, champ) for champ in role_dict['Support']])

x_cols = ['blueJungleChampTags', 'redJungleChampTags', 'blueMiddleChampTags', 'redMiddleChampTags', 'blueADCChampTags', 
            'redADCChampTags', 'blueSupportChampTags', 'redSupportChampTags', 'blueTopChampTags', 'redTopChampTags', 
            'rKills_pre15', 'rTowers_pre15', 'rDragons_pre15','rHeralds_pre15', 'golddiff_min15']

x = df[x_cols]
x = pd.get_dummies(x, columns=['blueJungleChampTags', 'redJungleChampTags', 'blueMiddleChampTags', 'redMiddleChampTags', 'blueADCChampTags', 
                                 'redADCChampTags', 'blueSupportChampTags', 'redSupportChampTags', 'blueTopChampTags', 'redTopChampTags'])

@app.route('/', methods=['GET', 'POST'])
def home():
    form = Form()
    return render_template('index.html', form=form)

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
    user_response = list(request.form.values())
    print('\n', '='*100, '\n')
    print(user_response)
    print('\n', '='*100, '\n')
    for i in range(10):
        print('\n', '='*100, '\n')
        print(user_response[i])
        print('\n', '='*100, '\n')
        user_response[i] = tag_dict[str(user_response[i].capitalize())]
        print('\n', '='*100, '\n')
        print(user_response)
        print('\n', '='*100, '\n')        
    
    features = [y+x if y != '' else int(x) for x, y in zip(user_response, ['redTopChampTags_', 'redJungleChampTags_', 'redMiddleChampTags_', 'redADCChampTags_', 'redSupportChampTags_', 
                                                                        'blueTopChampTags_', 'blueJungleChampTags_', 'blueMiddleChampTags_', 'blueADCChampTags_', 'blueSupportChampTags_', 
                                                                        '', '', '','', ''])]
    input_array = [0]*len(x.columns)

    print('\n', '='*100, '\n')
    for name in x.columns: 
        print(name)
    print('\n', '='*100, '\n')      

    print('\n', '='*100, '\n')
    for name in features: 
        print(name)
    print('\n', '='*100, '\n')    

    for i in range(len(x.columns)):
        if pd.Series.between(i, 11, 14):
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
    print('Your probability of winning is: {:.4%} for the input: {}'.format(output, features))
    return render_template('index.html', prediction_text='Your probability of winning is: {:.4%} for the input: {}'.format(output, features), form=Form())

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    :return:
    '''

if __name__ == "__main__":
    app.run(debug=True)

