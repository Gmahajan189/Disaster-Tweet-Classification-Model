import pickle
import re
import numpy as np

# Test example to load the vectorizer and model
try:
    # Load the vectorizer
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("Vectorizer loaded successfully.")

    # Load the model
    with open('disaster_tweet_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")

    # Check the type of the loaded model to ensure it's correct
    print(f"Loaded model type: {type(model)}")

except Exception as e:
    print(f"Error loading vectorizer or model: {e}")
    exit(1)

# Define text cleaning function
def clean_text(text):
    # Clean URLs, special characters, and digits, and convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove special characters and numbers
    text = re.sub(r'\d', '', text)  # Remove digits
    return text.lower()  # Convert to lowercase

# Sample tweet for testing
sample_tweet = "This is a test tweet with http://example.com"
print(f"Original Tweet: {sample_tweet}")

# Clean the tweet text
cleaned_tweet = clean_text(sample_tweet)
print(f"Cleaned Tweet: {cleaned_tweet}")

# Vectorizing the cleaned tweet
try:
    vectorized_tweet = vectorizer.transform([cleaned_tweet])  # Make sure it's wrapped in a list
    print(f"Vectorized Tweet Shape: {vectorized_tweet.shape}")
except Exception as e:
    print(f"Error during vectorization: {e}")
    exit(1)

# Predicting using the model
try:
    prediction = model.predict(vectorized_tweet)
    print(f"Prediction (raw): {prediction}")  # Output the prediction (raw)

    # If prediction is an array, print the first element (or map it to the corresponding label if it's a classification task)
    if isinstance(prediction, (list, np.ndarray)):
        print(f"Prediction (first element if array): {prediction[0]}")
    else:
        print(f"Prediction: {prediction}")

except Exception as e:
    print(f"Error during prediction: {e}")
    exit(1)
