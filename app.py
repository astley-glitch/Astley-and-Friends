from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('model/xgbModel.pkl','rb'))

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=["GET","POST"])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))

    if request.method == "POST":
        num_preg= request.form['num_preg']
        glucose_conc = request.form['glucose_conc']
        diastolic_bp = request.form['diastolic_bp']
        thickness = request.form['thickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diab_pred = request.form['diab_pred']
        age = request.form['age']
        skin = request.form['skin']

        input_variables = pd.DataFrame([[num_preg, glucose_conc, diastolic_bp, thickness, insulin,
        bmi, diab_pred, age, skin]],columns=['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin',
       'bmi', 'diab_pred', 'age', 'skin'])


        prediction = model.predict(input_variables)
        prob = model.predict_proba(input_variables)[0]
        factor = prob[0] *100
        confidence_factor = str(round(factor, 1)) +'%'
        if prediction == 0:
            result = 'YES'
        else:
            result = 'NO'
# 6	148	72	35	0	33.6	0.627	50	1.3790	
        return render_template('main.html',result=result,confidence_factor=confidence_factor)

    if __name__ == '__main__':
        app.run(debug=True)