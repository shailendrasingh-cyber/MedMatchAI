from flask import Flask, request, jsonify, render_template
import pickle
from sentence_transformers import SentenceTransformer, util
import pandas as pd

app = Flask(__name__)

# Load the model and dataframe
model = SentenceTransformer('saved_model/model')
df = pd.read_pickle('disease_symptoms.pkl')

# Function to find matching condition based on input symptoms
def find_condition_by_symptoms(input_symptoms):
    # Generate embedding for the input symptoms
    input_embedding = model.encode(input_symptoms)
    
    # Calculate similarity scores with each condition
    df['Similarity'] = df['Symptom_Embedding'].apply(lambda x: util.cos_sim(input_embedding, x).item())
    
    # Find the most similar condition
    best_match = df.loc[df['Similarity'].idxmax()]
    return best_match['Name'], best_match['Treatments']

# Flask route for the homepage (HTML form)
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_symptoms = data.get('symptoms', '')
        
        if not input_symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        condition_name, treatments = find_condition_by_symptoms(input_symptoms)
        return jsonify({'condition': condition_name, 'treatments': treatments})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
