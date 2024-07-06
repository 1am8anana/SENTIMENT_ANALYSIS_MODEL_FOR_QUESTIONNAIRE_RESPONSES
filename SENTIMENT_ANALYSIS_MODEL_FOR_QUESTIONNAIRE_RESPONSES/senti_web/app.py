from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the pre-trained model
with open('Classifier/Naive Bayes_pipeline.pickle', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        df = pd.read_csv(file)
        df['label'] = model.predict(df['sentence'])
        df['probability'] = df.apply(lambda row: model.predict_proba([row['sentence']])[0][row['label']], axis=1) # Correct probability
        table_html = df.head(10).to_html(index=False)
        df.to_csv('output.csv', index=False)
        return jsonify({'table': table_html})
    return jsonify({'error': 'File upload failed'})

@app.route('/download')
def download_file():
    path = 'output.csv'
    return send_file(path, as_attachment=True)

@app.route('/predict', methods=['POST'])
def predict_sentence():
    data = request.get_json()
    sentence = data['sentence']
    prediction = model.predict([sentence])[0]
    probability = model.predict_proba([sentence])[0][prediction]
    return jsonify({'prediction': int(prediction), 'probability': float(probability)})

if __name__ == '__main__':
    app.run(debug=True)
