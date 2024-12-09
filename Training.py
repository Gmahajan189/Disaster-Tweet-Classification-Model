import pickle
import re
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from Paramas import *  # Ensure this imports correct parameters

# Step 1: Load the data
# Assuming 'data' is a pandas DataFrame containing your dataset
# Make sure 'data' has columns 'text' and 'target'
# You can load your data here (e.g., data = pd.read_csv('your_data.csv'))

# Step 2: Group keywords into categories
def group_keywords(keyword):
    natural_disasters = ['flood', 'deluge', 'hurricane', 'earthquake', 'tsunami', 'forest%20fire', 'inundation']
    man_made_disasters = ['explosion', 'fire', 'collision', 'radiation%20emergency', 'armageddon']
    if keyword in natural_disasters:
        return 'natural_disaster'
    elif keyword in man_made_disasters:
        return 'man_made_disaster'
    else:
        return 'other'

data['keyword_category'] = data['keyword'].apply(group_keywords)

# Print the first few rows to check the 'keyword_category' column
print(data[['keyword', 'keyword_category']].head())

# Check the distribution of 'keyword_category' to see how the categories are distributed
print(data['keyword_category'].value_counts())

# Step 3: Extract the target (disaster vs non-disaster) and feature (tweet text)
X = data['text']  # assuming 'text' column contains the tweet content
y = data['keyword_category'].map({'other': 0, 'natural_disaster': 1, 'man_made_disaster': 1})  # Map categories to binary labels

# Step 4: Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Step 5: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 6: Apply SMOTE to balance the class distribution in the training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 7: Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)

# Step 8: Make predictions and evaluate the model
y_pred = rf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Step 9: Save the trained model and vectorizer
with open('disaster_tweet_model.pkl', 'wb') as file:
    pickle.dump(rf, file)  # Save the trained RandomForest model
print("Model saved as disaster_tweet_model.pkl")

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)  # Save the vectorizer
print("Vectorizer saved as vectorizer.pkl")
