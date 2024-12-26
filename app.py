from flask import Flask, render_template, request
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# API Details
API_URL = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}"}

# Function to query Hugging Face API
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Home route that handles both form display and result
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['text']

        # Query the API with user input
        output = query({"inputs": user_input})

        # Check if the output is as expected
        if isinstance(output, list) and len(output) > 0:
            sentiment_result = output[0]  # Extract the first element of the list
            labels = [item['label'] for item in sentiment_result]  # Get sentiment labels
            scores = [item['score'] for item in sentiment_result]  # Get sentiment scores
        else:
            # Handle the case where the output is not as expected
            labels = ['Error']
            scores = [0]

        # Send the results back to the same page
        return render_template('index.html', text=user_input, labels=labels, scores=scores)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
