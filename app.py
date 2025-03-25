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
    """
    Classify the sentiment of a review using GPT-4.
    
    Args:
        review_text (str): The review text to classify
        
    Returns:
        str: The sentiment classification ('positive' or 'negative')
    """
    try:
        # Create the prompt for sentiment classification
        prompt = f"""Classify the sentiment of the following review as either 'positive' or 'negative'.
        Review: {review_text}
        Sentiment:"""
        
        # Make API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Respond with only 'positive' or 'negative'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=10
        )
        
        # Extract and clean the response
        sentiment = response.choices[0].message.content.strip().lower()
        
        # Validate the response
        if sentiment not in ['positive', 'negative']:
            raise ValueError("Invalid sentiment classification received")
            
        return sentiment
        
    except Exception as e:
        raise Exception(f"Error in sentiment classification: {str(e)}")

@app.route('/classify', methods=['POST'])
def classify_review():
    """
    Endpoint to classify the sentiment of a review.
    
    Expected JSON payload:
    {
        "review": "Your review text here"
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate request data
        if not data or 'review' not in data:
            return jsonify({
                'error': 'Missing review text in request'
            }), 400
            
        review_text = data['review']
        
        # Validate review text
        if not isinstance(review_text, str) or not review_text.strip():
            return jsonify({
                'error': 'Invalid review text'
            }), 400
            
        # Classify sentiment
        sentiment = classify_sentiment(review_text)
        
        return jsonify({
            'review': review_text,
            'sentiment': sentiment
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return jsonify({
        'status': 'healthy'
    })

if __name__ == '__main__':
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Modify this to bind to the PORT environment variable for deployment
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
