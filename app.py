from flask import Flask, render_template, request, jsonify
import joblib
import os
import sys

app = Flask(__name__)

# Load the model and vectorizer with better error handling
model = None
vectorizer = None

def load_models():
    global model, vectorizer
    try:
        print("Current working directory:", os.getcwd())
        print("Looking for files:", os.listdir())
        
        print("Loading vectorizer...")
        vectorizer = joblib.load('vectorizer.pkl')
        print("Vectorizer type:", type(vectorizer))
        
        print("Loading model...")
        model = joblib.load('spam_model.pkl')
        print("Model type:", type(model))
        
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print("Full error:", sys.exc_info())
        return False

# Load models on startup
if not load_models():
    print("Warning: Models failed to load properly!")

def predict_spam(message):
    if model is None or vectorizer is None:
        return {'error': 'Models not loaded properly'}
    
    try:
        message = message.lower().strip()
        vec = vectorizer.transform([message])
        proba = model.predict_proba(vec)[0][1]  # Probability of spam
        prediction = model.predict(vec)[0]
        return {
            'prediction': 'SPAM' if prediction == 1 else 'HAM',
            'confidence': round(float(proba), 2)
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {'error': f'Prediction failed: {str(e)}'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        if not message:
            return jsonify({'error': 'Please enter a message'})
        
        result = predict_spam(message)
        return jsonify(result)

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run the app with production settings
    app.run(host='0.0.0.0', port=port) 