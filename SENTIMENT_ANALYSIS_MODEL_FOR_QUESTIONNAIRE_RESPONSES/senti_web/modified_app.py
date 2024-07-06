
from flask import Flask, request, render_template, send_file, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the Naive Bayes model
model_path = 'Naive Bayes_pipeline_tf.pickle'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        if 'sentence' not in df.columns:
            return jsonify({'error': 'CSV must contain a "sentence" column'})
        predictions = model.predict(df['sentence'])
        probabilities = model.predict_proba(df['sentence'])[:, 1]  # Probability of positive class
        df['label'] = predictions
        df['probability'] = probabilities
        sample_df = df.head(10)  # Display only first 10 rows
        sample_html = sample_df.to_html(classes="table table-striped", index=False)
        df.to_csv('labeled_sentences.csv', index=False)
        return jsonify({'table': sample_html})
    return jsonify({'error': 'Invalid file format'})

@app.route('/download')
def download():
    return send_file('labeled_sentences.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)


import streamlit as st
import pickle

# โหลดโมเดล Naive Bayes
with open('Naive Bayes_pipeline_tf.pickle', 'rb') as f:
    model = pickle.load(f)

# ฟังก์ชันสำหรับการพยากรณ์
def predict_sentiment(text):
    # ทำการพยากรณ์
    prediction = model.predict([text])
    
    # แปลงค่าพยากรณ์เป็นข้อความ
    sentiment = 'positive' if prediction == 1 else 'negative'
    return sentiment

# สร้างส่วนติดต่อผู้ใช้ด้วย Streamlit
st.title('Sentiment Analysis')

# แถบ input สำหรับรับข้อความจากผู้ใช้
user_input = st.text_area('Enter your text here:')

# ปุ่มสำหรับทำการพยากรณ์
if st.button('Predict'):
    if user_input:
        # ทำการพยากรณ์
        sentiment = predict_sentiment(user_input)
        st.write(f'The predicted sentiment is: {sentiment}')
    else:
        st.write('Please enter some text to predict.')
