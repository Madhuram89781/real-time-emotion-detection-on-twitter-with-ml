import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import re
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords
#madhuram
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def main():
    # Download the dataset
    print("Downloading dataset...")
    path = kagglehub.dataset_download("kazanova/sentiment140")
    print("Dataset downloaded to:", path)
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(f"{path}/training.1600000.processed.noemoticon.csv", 
                    encoding='latin-1', 
                    header=None,
                    names=['target', 'id', 'date', 'flag', 'user', 'text'])
    
    # Map target values (0 = negative, 4 = positive)
    df['target'] = df['target'].map({0: 0, 4: 1})
    
    # Preprocess the text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['target'], test_size=0.2, random_state=42
    )
    
    # Vectorize the text
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train the model
    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train_vec, y_train)
    test_score = model.score(X_test_vec, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Save the model and vectorizer
    print("Saving model and vectorizer...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Model and vectorizer saved successfully!")

if __name__ == "__main__":
    main() 