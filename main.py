from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model and vectorizer
try:
    with open('disaster_tweet_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

try:
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading vectorizer: {e}")
    exit(1)

# Define text cleaning function
def clean_text(text):
    # Clean URLs, special characters, and digits, and convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove special characters and numbers
    text = re.sub(r'\d', '', text)  # Remove digits
    return text.lower()  # Convert to lowercase

# Add this route for the root path
@app.route('/')
def home():
    return "Welcome to the Disaster Tweet Prediction API! Use the /predict endpoint to make predictions."

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the POST request
        data = request.get_json()

        # Ensure that 'tweet' is present in the request
        if 'tweet' not in data:
            return jsonify({'error': 'Tweet data is missing'}), 400

        tweet = data['tweet']

        # Clean the tweet text (lowercase and remove unwanted characters)
        cleaned_tweet = clean_text(tweet)

        # Vectorize the cleaned tweet (wrap it in a list)
        vectorized_tweet = vectorizer.transform([cleaned_tweet])

        # Predict using the trained model
        prediction = model.predict(vectorized_tweet)

        # Convert prediction to human-readable label
        label = "disaster" if prediction[0] == 1 else "other"

        # Return the prediction result
        return jsonify({'prediction': label})

    except Exception as e:
        # Log and return any errors that occur
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Running on port 5001
