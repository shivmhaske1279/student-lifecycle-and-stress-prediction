import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained SVM model
MODEL_PATH = 'SVM_pkl.pkl'
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Single HTML layout embedded using Tailwind CSS
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Prediction Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-slate-50 font-sans text-slate-800 min-h-screen flex flex-col justify-between">

    <header class="bg-gradient-to-r from-blue-600 to-indigo-700 text-white shadow-md py-6 px-4">
        <div class="max-w-4xl mx-auto text-center">
            <h1 class="text-3xl font-bold tracking-tight">Student Performance Analytics</h1>
            <p class="text-indigo-100 mt-2 text-sm md:text-base">Predictive modeling using Support Vector Machine (SVM)</p>
        </div>
    </header>

    <main class="max-w-4xl w-full mx-auto p-4 md:p-8 flex-grow">
        
        {% if prediction_text %}
        <div class="mb-6 p-4 bg-emerald-50 border-l-4 border-emerald-500 rounded shadow-sm text-center">
            <h2 class="text-xl font-bold text-emerald-800">{{ prediction_text }}</h2>
        </div>
        {% endif %}

        {% if error_text %}
        <div class="mb-6 p-4 bg-rose-50 border-l-4 border-rose-500 rounded shadow-sm">
            <h2 class="text-lg font-medium text-rose-800">{{ error_text }}</h2>
        </div>
        {% endif %}

        <div class="bg-white rounded-xl shadow-md border border-slate-100 p-6 md:p-8">
            <h2 class="text-xl font-semibold text-slate-700 mb-6 border-b pb-2 border-slate-100">Input Metrics</h2>
            
            <form action="/predict" method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <div>
                    <label class="block text-sm font-semibold text-slate-600 mb-1">Student Type (Numeric Code)</label>
                    <input type="number" step="any" name="Student_Type" required 
                           value="{{ form_data.Student_Type if form_data else '1' }}"
                           class="w-full px-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-slate-600 mb-1">Daily Sleep Hours</label>
                    <input type="number" step="any" name="Sleep_Hours" required 
                           value="{{ form_data.Sleep_Hours if form_data else '7' }}"
                           class="w-full px-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-slate-600 mb-1">Daily Study Hours</label>
                    <input type="number" step="any" name="Study_Hours" required 
                           value="{{ form_data.Study_Hours if form_data else '4' }}"
                           class="w-full px-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-slate-600 mb-1">Social Media Hours</label>
                    <input type="number" step="any" name="Social_Media_Hours" required 
                           value="{{ form_data.Social_Media_Hours if form_data else '2' }}"
                           class="w-full px-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-slate-600 mb-1">Attendance (%)</label>
                    <input type="number" step="any" name="Attendance" required min="0" max="100"
                           value="{{ form_data.Attendance if form_data else '85' }}"
                           class="w-full px-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-slate-600 mb-1">Exam Pressure Score</label>
                    <input type="number" step="any" name="Exam_Pressure" required 
                           value="{{ form_data.Exam_Pressure if form_data else '3' }}"
                           class="w-full px-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-slate-600 mb-1">Family Support Score</label>
                    <input type="number" step="any" name="Family_Support" required 
                           value="{{ form_data.Family_Support if form_data else '4' }}"
                           class="w-full px-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-slate-600 mb-1">Month (1-12)</label>
                    <input type="number" name="Month" required min="1" max="12"
                           value="{{ form_data.Month if form_data else '6' }}"
                           class="w-full px-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                </div>

                <div class="md:col-span-2 mt-4">
                    <button type="submit" 
                            class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg shadow transition duration-150 ease-in-out cursor-pointer text-center">
                        Generate Prediction
                    </button>
                </div>
            </form>
        </div>
    </main>

    <footer class="bg-slate-800 text-slate-400 text-center py-4 text-xs border-t border-slate-700">
        &copy; 2026 Machine Learning Deployment Dashboard
    </footer>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_LAYOUT)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data to pass back into features array
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
        
        # Format for Scikit-Learn SVM prediction
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        
        return render_template_string(
            HTML_LAYOUT, 
            prediction_text=f'Prediction Result: {prediction[0]}',
            form_data=request.form
        )
        
    except Exception as e:
        return render_template_string(HTML_LAYOUT, error_text=f'Error in prediction: {str(e)}')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
