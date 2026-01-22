from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join('model', 'house_price_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from form
            overall_qual = int(request.form['overall_qual'])
            gr_liv_area = int(request.form['gr_liv_area'])
            total_bsmt_sf = int(request.form['total_bsmt_sf'])
            garage_cars = int(request.form['garage_cars'])
            full_bath = int(request.form['full_bath'])
            year_built = int(request.form['year_built'])

            # Create DataFrame for model
            input_data = pd.DataFrame({
                'OverallQual': [overall_qual],
                'GrLivArea': [gr_liv_area],
                'TotalBsmtSF': [total_bsmt_sf],
                'GarageCars': [garage_cars],
                'YearBuilt': [year_built],
                'FullBath': [full_bath]
            })

            # Predict
            prediction = model.predict(input_data)[0]

            return render_template('index.html', prediction=f"${prediction:,.2f}")

        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
