import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle
import os

# Load the dataset
df = pd.read_csv('Diseases_Symptoms.csv')

# Initialize a Sentence Transformer model to generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each condition's symptoms
df['Symptom_Embedding'] = df['Symptoms'].apply(lambda x: model.encode(x))

# Define the directories where files will be saved
model_dir = 'saved_model'
pkl_file = 'disease_symptoms.pkl'
combined_file = 'model_and_data.pkl'

# Create directories if they don't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the SentenceTransformer model using the model.save() method
model.save(os.path.join(model_dir, 'model'))

# Save the precomputed embeddings and dataframe using to_pickle()
df.to_pickle(pkl_file)

# Additionally, save the model and dataframe using pickle for later reuse
with open(combined_file, 'wb') as file:
    pickle.dump({'model': model, 'df': df}, file)

# Function to find matching condition based on input symptoms
def find_condition_by_symptoms(input_symptoms):
    # Generate embedding for the input symptoms
    input_embedding = model.encode(input_symptoms)
    
    # Calculate similarity scores with each condition
    df['Similarity'] = df['Symptom_Embedding'].apply(lambda x: util.cos_sim(input_embedding, x).item())
    
    # Find the most similar condition
    best_match = df.loc[df['Similarity'].idxmax()]
    return best_match['Name'], best_match['Treatments']

# Example input to test the functionality
input_symptoms = "Sweating, Trembling, Fear of losing control"
condition_name, treatments = find_condition_by_symptoms(input_symptoms)

# Print out the result
print("Condition:", condition_name)
print("Recommended Treatments:", treatments)
