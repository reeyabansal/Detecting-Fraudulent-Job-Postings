# src/preprocess.py

import pandas as pd
import re

def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Replace non-word characters with space
    text = text.lower()  # Convert to lowercase
    return text

def load_and_clean_data(filename1, filename2):
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)

    df = pd.concat([df1, df2])
    
    df.dropna(inplace=True)  # Drop rows with NaN values
    
    for col in ['title', 'description', 'requirements', 'location', 'company_profile']:
        df[col] = df[col].apply(clean_text)

    df['combined_features'] = (df['title'] + ' ' +
                               df['description'] + ' ' +
                               df['requirements'] + ' ' +
                               df['location'] + ' ' +
                               df['company_profile'])

    return df[['combined_features', 'fraudulent']]
