import numpy as np
import pandas as pd
from Paramas import *
import pandas as pd
import seaborn as sns
import re
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords

# Handle missing values
data['keyword'].fillna('unknown', inplace=True)  # Replace missing keywords with 'unknown'
data.drop(columns=['location'], inplace=True)  # Drop 'location' due to high sparsity


def clean_text(text):
    print(f"Original text: {text}")  # Print the original text
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    print(f"After URL removal: {text}")  # Print after URL removal

    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    print(f"After removing special characters: {text}")  # Print after removing special characters

    # Remove digits
    text = re.sub(r'\d', '', text)
    print(f"After removing digits: {text}")  # Print after removing digits
    
    # Convert text to lowercase
    cleaned_text = text.lower()
    print(f"Final cleaned text: {cleaned_text}")  # Print the final cleaned text
    
    return cleaned_text

# Clean the text column
data['text'] = data['text'].apply(clean_text)

# Display the cleaned dataset
print("Cleaned Data Sample:")
print(data.head())

# Check for missing values after preprocessing
print("\nMissing Values Summary:")
print(data.isnull().sum())

