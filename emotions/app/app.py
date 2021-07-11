import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import tensorflow_text
from flask import Flask, request, json
from flask_cors import CORS
from api import get_youtube_data, parse_data


# Load model
print('Loading model...')
model = tf.keras.models.load_model('./emotion_model.h5', custom_objects={ 'auc': tf.keras.metrics.AUC, 'f1': tfa.metrics.F1Score, 'cohenKappa': tfa.metrics.CohenKappa })
print('Model loaded.')

# Init flask app
app = Flask(__name__)
# Enable CORS
CORS(app)


# Make prediction with model
def make_prediction(input_sentence):
    # Make prediction with model
    print('Elaborating input...')

    input_sentence = np.array([input_sentence])
    result = model.predict(input_sentence)

    print('Done')

    # Translate prediction
    decode_map = {
        0: 'constructive feedback/idea',
        1: 'negative',
        2: 'neutral/other', 
        3: 'positive', 
        4: 'sadness', 
    }

    return decode_map[np.argmax(result)]


# Main route set up
@app.route('/')
def home():
    if request.method == 'GET':
        return 'Welcome! Now you can make new predictions!'



# Route for predicting only 1 comment
@app.route('/api/comment', methods=['POST'])
def predict_comment():
    if request.method == 'POST':
        # Get data and process it
        input_data = str(request.get_json()['data'])
        
        # Get prediction
        result = make_prediction(input_data)
        
        data = {
            'success': True,
            'comment': input_data,
            'prediction': result
        }
        print(data)
        # Return predictions
        response = app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json'
        )
        return response


# Route for predicting multiple comments
# URL = /api/comments/<video_ID>?maxResults=[...]&order=[...]
@app.route('/api/comments/<video_ID>', methods=['POST'])
def predict_comments(video_ID):
    # Get query parameters
    max_results = request.args.get('maxResults')
    order = request.args.get('order')

    try:
        # Make request to API
        result = get_youtube_data(video_ID=video_ID, max_results=(max_results or 20), order=(order or 'relevance'))
        # Check result of request
        if result['success'] == False: return result
        # Parse returned data
        comments = parse_data(result['data'])
        # Make predictions
        for idx, comment in enumerate(comments):
            result = make_prediction(comment['comment'])
            comments[idx]['prediction'] = result
        # Return success response
        return {
            'success': True,
            'numberOfResults': len(comments),
            'data': comments
        }
    except:
        # Return error responses
        return {
            'success': False,
            'message': 'Something went wrong'
        }
