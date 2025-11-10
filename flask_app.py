from flask import Flask, request, jsonify
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / '..' / 'models' / 'final_logistic_model.joblib')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
	data = request.get_json(force=True)
	try:
		# preserve feature ordering used during training
		values = [data[k] for k in sorted(data.keys())]
		proba = model.predict_proba([values])[0][1]
		pred = int(proba >= 0.5)
		return jsonify({'churn_probability': float(proba), 'prediction': pred})
	except Exception as e:
		return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080)
