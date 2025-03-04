import json
import joblib
import numpy as np

model = joblib.load('/var/task/churn_model.pkl')
encoder = joblib.load('/var/task/encoder.pkl')
scaler = joblib.load('/var/task/scaler.pkl')

def lambda_handler(event, context):
    try:
        if 'body' not in event or event['body'] is None:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Missing request body'})}

        body = json.loads(event['body'])  # Extracting JSON from the request
        if 'features' not in body:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Missing "features" field'})}

        try:
            data_array = np.array([float(x) for x in body['features']]).reshape(1, -1)
        except ValueError:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid input type: All values must be numbers'})}

        prediction = model.predict(data_array)

        return {
            'statusCode': 200,
            'body': json.dumps({'churn_prediction': int(prediction[0])})
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
