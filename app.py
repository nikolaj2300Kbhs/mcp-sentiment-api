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

def predict_box_score(historical_data, future_box_info):
    """Simulate a 1–10 satisfaction score for a future box using historical data."""
    try:
        prompt = f"""You are a Goodiebox satisfaction expert simulating a member satisfaction score (1–10) for a future subscription box. Use this data context:

        **Data Explanation**:
        - Historical Data: Past boxes with details like:
          - Box SKU: Unique box identifier (e.g., DK-2504-CLA-2L).
          - Products: Number of items, listed as Product SKUs (e.g., SKU123).
          - Total Retail Value: Sum of product retail prices in €.
          - Unique Categories: Number of distinct product categories (e.g., skincare, makeup).
          - Full-size/Premium: Counts of full-size items and those >€20.
          - Total Weight: Sum of product weights in grams.
          - Avg Brand/Category Ratings: Average ratings (out of 5) for brands and categories.
          - Historical Score: Past average box rating (out of 5).
          - Past Predicted Score: Previous predictions (1–10).
        - Future Box Info: Details of a new box (same format, no historical score yet).

        **Inputs**:
        Historical Data (past boxes): {historical_data}
        Future Box Info: {future_box_info}

        Simulate the score by analyzing trends in past member reactions, product variety, retail value, brand reputation, category ratings, and surprise value. Return only a number (1–10)."""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You’re an expert in predicting Goodiebox satisfaction, skilled at simulating outcomes from historical trends."},
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
        raise Exception(f"Error in box score simulation: {str(e)}")

@app.route('/predict_box_score', methods=['POST'])
def box_score():
    """Endpoint for simulating future box scores."""
    try:
        data = request.get_json()
        if not data or 'future_box_info' not in data:
            return jsonify({'error': 'Missing future box info'}), 400
        historical_data = data.get('historical_data', 'No historical data provided')
        future_box_info = data['future_box_info']
        score = predict_box_score(historical_data, future_box_info)
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
