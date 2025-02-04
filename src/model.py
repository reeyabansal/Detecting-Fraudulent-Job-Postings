# src/model.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Replace non-word characters with space
    text = text.lower()  # Convert to lowercase
    return text

def data_loading_preprocessing(filename1, filename2):
    dataframe1 = pd.read_csv(filename1)
    dataframe2 = pd.read_csv(filename2)

    dataframe = pd.concat([dataframe1, dataframe2])
    dataframe = dataframe[['title', 'description', 'requirements', 'location', 'company_profile', 'fraudulent']]
    
    # Drop NaN rows
    dataframe.dropna(axis=0, how='any', inplace=True)

    # Clean Text - Applied to relevant columns
    dataframe['title'] = dataframe['title'].apply(clean_text)
    dataframe['description'] = dataframe['description'].apply(clean_text)
    dataframe['requirements'] = dataframe['requirements'].apply(clean_text)
    dataframe['location'] = dataframe['location'].apply(clean_text)
    dataframe['company_profile'] = dataframe['company_profile'].apply(clean_text)

    dataframe['combined_features'] = (dataframe['title'] + ' ' +
                                       dataframe['description'] + ' ' +
                                       dataframe['requirements'] + ' ' +
                                       dataframe['location'] + ' ' +
                                       dataframe['company_profile'])
    
    return dataframe

def feature_extraction(dataframe: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=5000)
    joblib.dump(vectorizer, 'vectorizer.pkl')
    X = vectorizer.fit_transform(dataframe['combined_features']).toarray()
    y = dataframe['fraudulent']
    
    return X, y

def train_model(file1, file2):
    jobs_dataframe = data_loading_preprocessing(file1, file2)
    X, y = feature_extraction(jobs_dataframe)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Hyperparameter tuning
    param_grid = {'C': [0.1, 1, 10]}
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'fake_job_detector.pkl')

if __name__ == '__main__':
    train_model('Fake Postings.csv', 'fake_job_postings.csv')
