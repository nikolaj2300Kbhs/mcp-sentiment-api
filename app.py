from flask import Flask, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def classify_sentiment(review_text):
    """Classify sentiment of a single review."""
    try:
        prompt = f"""Classify the sentiment of the following review as 'positive' or 'negative'.
        Review: {review_text}
        Sentiment:"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Respond with only 'positive' or 'negative'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=10
        )
        sentiment = response.choices[0].message.content.strip().lower()
        if sentiment not in ['positive', 'negative']:
            raise ValueError("Invalid sentiment classification")
        return sentiment
    except Exception as e:
        raise Exception(f"Error in sentiment classification: {str(e)}")

def predict_box_score(reviews, box_info):
    """Predict a 1–10 satisfaction score for a box based on reviews and box info."""
    try:
        prompt = f"""Predict a satisfaction score (1–10) for a Goodiebox subscription box based on these inputs:
        Reviews: {', '.join(reviews)}
        Box Info: {box_info}
        Consider factors like sentiment, product variety, retail value, and surprise value. Return only a number (1–10)."""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Goodiebox satisfaction expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=5
        )
        score = response.choices[0].message.content.strip()
        if not score.isdigit() or int(score) < 1 or int(score) > 10:
            raise ValueError("Invalid score received")
        return score
    except Exception as e:
        raise Exception(f"Error in box score prediction: {str(e)}")

@app.route('/sentiment', methods=['POST'])
def classify_review():
    """Existing endpoint for sentiment classification."""
    try:
        data = request.get_json()
        if not data or 'review' not in data:
            return jsonify({'error': 'Missing review text'}), 400
        review_text = data['review']
        if not isinstance(review_text, str) or not review_text.strip():
            return jsonify({'error': 'Invalid review text'}), 400
        sentiment = classify_sentiment(review_text)
        return jsonify({'review': review_text, 'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_box_score', methods=['POST'])
def box_score():
    """New endpoint to predict box satisfaction score."""
    try:
        data = request.get_json()
        if not data or 'reviews' not in data:
            return jsonify({'error': 'Missing reviews'}), 400
        reviews = data['reviews']
        box_info = data.get('box_info', 'No additional info provided')
        if not isinstance(reviews, list) or not all(isinstance(r, str) for r in reviews):
            return jsonify({'error': 'Reviews must be a list of strings'}), 400
        score = predict_box_score(reviews, box_info)
        return jsonify({'predicted_box_score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
