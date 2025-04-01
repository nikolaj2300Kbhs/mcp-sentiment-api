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

def predict_box_score(reviews, box_info, historical_data):
    """Predict a 1–10 satisfaction score with detailed inputs."""
    try:
        prompt = f"""Predict a satisfaction score (1–10) for a Goodiebox subscription box based on:
        Reviews: {', '.join(reviews) if reviews else 'No reviews provided'}
        Box Info: {box_info}
        Historical Data: {historical_data}
        Evaluate based on:
        - Sentiment from reviews (40% weight): Positive/negative tone.
        - Product variety (30% weight): Number of products, unique categories.
        - Retail value and surprise (20% weight): Total value, full-size/premium items.
        - Historical trends (10% weight): Past scores/reactions.
        Return only a number (1–10)."""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You’re a Goodiebox satisfaction expert with deep knowledge of beauty box trends."},
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

@app.route('/predict_box_score', methods=['POST'])
def box_score():
    """Endpoint for box score prediction."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing data'}), 400
        reviews = data.get('reviews', [])
        box_info = data.get('box_info', 'No additional info provided')
        historical_data = data.get('historical_data', 'No historical data provided')
        score = predict_box_score(reviews, box_info, historical_data)
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
