from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
app = Flask(__name__)

# Load your trained models
best_rf = joblib.load('models/best_rf_ctt_fp.pkl')
best_model = joblib.load('models/best_model_fp_fsi.pkl')
columns1 = pd.Index([
    'DevelopmentUnit_Cardiovascular', 'DevelopmentUnit_NeuroScience',
    'DevelopmentUnit_Oncology', 'DevelopmentUnit_Respiratory',
    'Phase_Phase I', 'Phase_Phase II', 'Phase_Phase III', 'Phase_Phase IV',
    'New Indication_No', 'New Indication_Yes',
    'Blinding_Double Blind', 'Blinding_Open Label', 'Blinding_Single Blind',
    'PediatricOnly_No', 'PediatricOnly_Yes'
])

columns2 = pd.Index([
    'Country_Argentina', 'Country_Australia', 'Country_Brazil', 'Country_Canada',
    'Country_China', 'Country_France', 'Country_India', 'Country_Italy',
    'Country_Japan', 'Country_South Africa', 'Country_Spain', 'Country_UK',
    'Country_USA', 'DevelopmentUnit_Cardiovascular', 'DevelopmentUnit_NeuroScience',
    'DevelopmentUnit_Oncology', 'DevelopmentUnit_Respiratory',
    'Phase_Phase I', 'Phase_Phase II', 'Phase_Phase III', 'Phase_Phase IV',
    'New Indication_No', 'New Indication_Yes',
    'Blinding_Double Blind', 'Blinding_Open Label', 'Blinding_Single Blind',
    'PediatricOnly_No', 'PediatricOnly_Yes'
])
@app.route('/')
def main():
    return render_template('mainpage.html')




def convert_to_array(user_input, columns):
    input_array = [0] * len(columns)
    for key, value in user_input.items():
        column_name = f"{key}_{value}"
        if column_name in columns:
            index = columns.get_loc(column_name)
            input_array[index] = 1
    return input_array

@app.route('/index', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        user_input = {
            'Country': request.form['country'],
            'DevelopmentUnit': request.form['development_unit'],
            'Phase': request.form['phase'],
            'New Indication': request.form['new_indication'],
            'Blinding': request.form['blinding'],
            'Pediatric': request.form['pediatric']
        }

        # Create separate dictionaries for CTT-FP and FP-FSI
        user_input_CTT_FP = {
            'DevelopmentUnit': user_input['DevelopmentUnit'],
            'Phase': user_input['Phase'],
            'New Indication': user_input['New Indication'],
            'Blinding': user_input['Blinding'],
            'Pediatric': user_input['Pediatric']
        }

        user_input_FP_FSI = {
            'Country': user_input['Country'],
            'DevelopmentUnit': user_input['DevelopmentUnit'],
            'Phase': user_input['Phase'],
            'New Indication': user_input['New Indication'],
            'Blinding': user_input['Blinding'],
            'Pediatric': user_input['Pediatric']
        }

        # Convert user input to arrays
        input_array1 = convert_to_array(user_input_CTT_FP, columns1)
        input_array2 = convert_to_array(user_input_FP_FSI, columns2)

        # Convert the 1D arrays to 2D arrays
        input_2d_array1 = np.array(input_array1).reshape(1, -1)
        input_2d_array2 = np.array(input_array2).reshape(1, -1)

        # Predict the weeks
        y_pred1 = best_rf.predict(input_2d_array1)[0]
        y_pred2 = best_model.predict(input_2d_array2)[0]
        total_weeks = y_pred1 + y_pred2

        result = {
            'ctt_fp': f"{y_pred1:.2f}",
            'fp_fsi': f"{y_pred2:.2f}",
            'total_weeks': f"{total_weeks:.2f}"
        }

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
