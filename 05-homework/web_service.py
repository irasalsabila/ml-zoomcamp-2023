from flask import Flask
from flask import jsonify,request
import pickle
from waitress import serve

def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


dv = load('dv.bin')
model = load('model1.bin')


app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]

    return jsonify({'prediction': float(y_pred),
                    "credit": bool(y_pred >= 0.5)})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)   
