import os
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the trained SVM model
MODEL_PATH = 'SVM_pkl.pkl'
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the HTML form post
        # Mapping inputs to numerical format expected by the model
        features = [
            float(request.form.get('Student_Type', 0)),
            float(request.form.get('Sleep_Hours', 0)),
            float(request.form.get('Study_Hours', 0)),
            float(request.form.get('Social_Media_Hours', 0)),
            float(request.form.get('Attendance', 0)),
            float(request.form.get('Exam_Pressure', 0)),
            float(request.form.get('Family_Support', 0)),
            float(request.form.get('Month', 1))
        ]
        
        # Convert to numpy array and reshape for prediction
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        
        # Format output message
        output = prediction[0]
        
        return render_template(
            'index.html', 
            prediction_text=f'Prediction Result: {output}',
            form_data=request.form
        )
        
    except Exception as e:
        return render_template('index.html', error_text=f'Error in prediction: {str(e)}')

if __name__ == "__main__":
    # Render binds to the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
