# This is the app
'''

---- 1 -----

Certain champions are not working well because of the spelling:
(Champion role spelling)                        (Champion tag spelling)
Cho'Gath                                            Chogath
Rek'Sai                                             Reksai
MissFortune                                         Miss Fortune
Monkeyking (Wukong)                                 Wukong

---- 2 -----

Create actual homepage
fix aesthetics on model pages
add back button for predict route 

Names need to be unified without spaces, and capitalized in order to standarize, wukong being monkey king is just odd
pog
'''
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, session
import pickle
import ast
import json
from flask_wtf import FlaskForm
from wtforms import SelectField
from sklearn.preprocessing import StandardScaler


app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secret'

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

@app.route('/', methods=['GET', 'POST'])
def home():
    print(session)
    form = Form()
    return render_template('index.html', form=form)

@app.route('/predict', methods=['POST'])
def predict():
    if session.get('my_var', None) == 'M1':
        model = pickle.load(open('models/rForestM1.pkl', 'rb'))
        
        x_cols = ['blueJungleChampTags', 'redJungleChampTags', 'blueMiddleChampTags', 'redMiddleChampTags', 'blueADCChampTags', 
        'redADCChampTags', 'blueSupportChampTags', 'redSupportChampTags', 'blueTopChampTags', 'redTopChampTags', 
        'rKills_pre15', 'rTowers_pre15', 'rDragons_pre15','rHeralds_pre15'] # , 'golddiff_min15'
        
        x = df[x_cols]
        x = pd.get_dummies(x, columns=['blueJungleChampTags', 'redJungleChampTags', 'blueMiddleChampTags', 'redMiddleChampTags', 'blueADCChampTags', 
                                 'redADCChampTags', 'blueSupportChampTags', 'redSupportChampTags', 'blueTopChampTags', 'redTopChampTags'])
        
        user_response = list(request.form.values())
        
        for i in range(10):
            user_response[i] = tag_dict[str(user_response[i].capitalize())]
        
        features = [y+x if y != '' else int(x) for x, y in zip(user_response, ['redTopChampTags_', 'redJungleChampTags_', 'redMiddleChampTags_', 'redADCChampTags_', 'redSupportChampTags_', 
                                                                            'blueTopChampTags_', 'blueJungleChampTags_', 'blueMiddleChampTags_', 'blueADCChampTags_', 'blueSupportChampTags_', 
                                                                            '', '', '',''])] # add another empty
        
        input_array = [0]*len(x.columns)

        for index, index2 in zip(range(4), range(10,14)): # put back to 15
            input_array[index] = features[index2]

        for i in range(len(x.columns)):
            if pd.Series.between(i, 0, 3): # put back to 4
                pass
            elif x.columns[i] in features:
                print('champ {} in {}'.format(x.columns[i], i))
                input_array[i] = 1    
            else:
                input_array[i] = 0

        prediction = model.predict_proba([np.array(input_array)])
        output = round(prediction[0][1], 3)
        print('Your probability of winning is: {:.4%} for the input: {}'.format(output, features))
        return render_template('models.html', prediction_text='Your probability of winning is: {:.3%}'.format(output), form=Form())
    else:
        model = pickle.load(open('models/rForestM2.pkl', 'rb'))
        
        x_cols = ['blueJungleChampTags', 'redJungleChampTags', 'blueMiddleChampTags', 'redMiddleChampTags', 'blueADCChampTags', 
        'redADCChampTags', 'blueSupportChampTags', 'redSupportChampTags', 'blueTopChampTags', 'redTopChampTags', 
        'rKills_pre15', 'rTowers_pre15', 'rDragons_pre15','rHeralds_pre15', 'golddiff_min15'] 
        
        x = df[x_cols]
        x = pd.get_dummies(x, columns=['blueJungleChampTags', 'redJungleChampTags', 'blueMiddleChampTags', 'redMiddleChampTags', 'blueADCChampTags', 
                                 'redADCChampTags', 'blueSupportChampTags', 'redSupportChampTags', 'blueTopChampTags', 'redTopChampTags'])
        
        user_response = list(request.form.values())
        print(user_response)
        for i in range(10):
            user_response[i] = tag_dict[str(user_response[i].capitalize())]
        
        features = [y+x if y != '' else int(x) for x, y in zip(user_response, ['redTopChampTags_', 'redJungleChampTags_', 'redMiddleChampTags_', 'redADCChampTags_', 'redSupportChampTags_', 
                                                                            'blueTopChampTags_', 'blueJungleChampTags_', 'blueMiddleChampTags_', 'blueADCChampTags_', 'blueSupportChampTags_', 
                                                                            '', '', '','', ''])]
        
        input_array = [0]*len(x.columns)

        for index, index2 in zip(range(5), range(10,15)):
            input_array[index] = features[index2]

        for i in range(len(x.columns)):
            if pd.Series.between(i, 0, 4):
                pass
            elif x.columns[i] in features:
                print('champ {} in {}'.format(x.columns[i], i))
                input_array[i] = 1    
            else:
                input_array[i] = 0

        prediction = model.predict_proba([np.array(input_array)])
        output = round(prediction[0][0], 3)
        print('Your probability of winning is: {:.4%} for the input: {}'.format(output, features))
        return render_template('models.html', prediction_text='Your probability of winning is: {:.3%}'.format(output), form=Form())

@app.route('/Model1', methods=['POST', 'GET'])
def Model1():
    isIndex=False
    session['my_var'] = 'M1'
    form = Form()
    return render_template('models.html', form=form, isIndex=isIndex)

@app.route('/Model2', methods=['POST', 'GET'])
def Model2():
    isIndex=True
    session['my_var'] = 'M2'
    form = Form()
    return render_template('models.html', form=form, isIndex=isIndex)


if __name__ == "__main__":
    app.run(debug=True)

