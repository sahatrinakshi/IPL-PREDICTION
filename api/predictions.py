import json
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def handler(request):
    try:
        path = os.path.join(MODEL_DIR, 'predictions.json')
        if not os.path.exists(path):
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Model not trained'})
            }

        with open(path) as f:
            data = json.load(f)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(data)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
