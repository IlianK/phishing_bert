from flask import Flask, json, request, jsonify, render_template
import os
import sys
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.helper_functions.path_resolver import DynamicPathResolver
from src.XAI import explain_bert

# -------------------------------------------------------------------
# Paths & Models
# -------------------------------------------------------------------

dpr = DynamicPathResolver(marker="README.md")
output_dir = dpr.path.models.bert._path
model_folder = os.path.join(output_dir, "checkpoint-2500")

explanations_json_path = dpr.path.app.static.json.lime_explanations_json
emails_path = dpr.path.app.static.csv.emails_csv
emails_df = pd.read_csv(emails_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, model = explain_bert.load_model(model_folder, device)
explanations_data = explain_bert.load_explanations(explanations_json_path)

# -------------------------------------------------------------------

app = Flask(__name__)

# -------------------------------------------------------------------
# Routes index.html
# -------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    subject = data.get("subject", "").strip()
    body = data.get("body", "").strip()

    if not subject and not body:
        return jsonify({"error": "Both subject and body cannot be empty."}), 400

    combined_text = subject + " [SEP] " + body

    # Predict Label
    predicted_label, confidence = explain_bert.predict_label(combined_text, tokenizer, model)

    # LIME Explanation
    lime_html = explain_bert.explain_prediction(combined_text, tokenizer, model)

    return jsonify({
        "predicted_label": predicted_label,
        "confidence": confidence,
        "lime_html": lime_html
    })


# -------------------------------------------------------------------
# Routes for learn_phishing.html
# -------------------------------------------------------------------
@app.route('/learn_phishing')
def learn_phishing():
    return render_template('learn_phishing.html')


@app.route('/get_email', methods=['POST'])
def get_email():
    req = request.get_json()
    email_index = req.get('email_index', 0)
    if 0 <= email_index < len(explanations_data):
        entry = explanations_data[email_index]
        return jsonify({
            'subject': entry.get('original_subject', ''),
            'body': entry.get('original_body', '')
        })
    return jsonify({'error': 'Invalid email index'}), 400


@app.route('/get_email_label', methods=['POST'])
def get_email_label():
    req = request.get_json()
    email_index = req.get('email_index', 0)
    if 0 <= email_index < len(explanations_data):
        entry = explanations_data[email_index]
        return jsonify({'label': entry.get('label', 0)})
    return jsonify({'error': 'Invalid email index'}), 400


@app.route('/get_lime_html', methods=['POST'])
def get_lime_html():
    req = request.get_json()
    email_index = req.get('email_index', 0)
    if 0 <= email_index < len(explanations_data):
        entry = explanations_data[email_index]
        return jsonify({'lime_html': entry.get('lime_html', '')})
    return jsonify({'error': 'Invalid email index'}), 400


if __name__ == '__main__':
    app.run(debug=True)
