from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Load email data
emails_path = os.path.join(os.getcwd(), 'app', 'static', 'csv', 'emails.csv')
emails_df = pd.read_csv(emails_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learn_phishing')
def learn_phishing():
    return render_template('learn_phishing.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Placeholder for integration with BERT and XAI later
    subject = request.json.get('subject', '')
    body = request.json.get('body', '')
    return jsonify({
        'xai_results': f"Analysis for subject: {subject} and body: {body}",
    })

@app.route('/get_email', methods=['POST'])
def get_email():
    # Fetch email content by index
    email_index = request.json.get('email_index', 0)
    if 0 <= email_index < len(emails_df):
        email = emails_df.iloc[email_index].to_dict()
        return jsonify(email)
    return jsonify({'error': 'Invalid email index'}), 400

@app.route('/get_email_label', methods=['POST'])
def get_email_label():
    # Fetch the email index from the request
    email_index = request.json.get('email_index', 0)
    
    # Ensure the index is valid
    if 0 <= email_index < len(emails_df):
        # Get the label for the email (assuming it's stored in the 'label' column)
        email_label = int(emails_df.iloc[email_index]['label'])  # Convert to native int
        return jsonify({'label': email_label})
    
    return jsonify({'error': 'Invalid email index'}), 400


if __name__ == '__main__':
    app.run(debug=True)
