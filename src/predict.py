# src/predict.py

import joblib
import pandas as pd
from preprocess import clean_text

def load_model(model_path='fake_job_detector.pkl'):
    return joblib.load(model_path)

def create_combined_features():
    """Create combined features from individual text fields."""
    title = clean_text(title)
    description = clean_text(description)
    requirements = clean_text(requirements)
    location = clean_text(location)
    company_profile = clean_text(company_profile)
    
    combined_features = f"{title} {description} {requirements} {location} {company_profile}"
    
    return combined_features

def predict(title, description, requirements, location, company_profile):
    model = load_model()
    
    # Assuming new_data is a DataFrame with a column named "combined_features"
    # Create combined features
    combined_features = create_combined_features(title, description, requirements, location, company_profile)

    # Vectorization (assuming the same vectorizer used during training is available)
    vectorizer = joblib.load('vectorizer.pkl')  # Load the vectorizer if saved separately
    new_data = pd.DataFrame({'combined_features': [combined_features]})
    
    # Transform the combined features into the same format as training data
    new_features = vectorizer.transform(new_data['combined_features']).toarray()

    predictions = model.predict(new_features)
    
    return predictions[0]

if __name__ == '__main__':
   title = input('Enter Title: ')
   description = input('Enter Description: ')
   requirements = input('Enter Requirements: ')
   location = input('Enter Location:' )
   company_profile = input('Enter Company Profile: ')

   result = predict(title, description, requirements, location, company_profile)
   print("Prediction (1 for fraudulent, 0 for not fraudulent):", result)
