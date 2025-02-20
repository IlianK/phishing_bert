from flask import Flask, json, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import lime.lime_text
import numpy as np
import pandas as pd
import os,sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.helper_functions.path_resolver import DynamicPathResolver

from transformers import AutoTokenizer, AutoModelForSequenceClassification

################################################################################

dpr = DynamicPathResolver(marker="README.md")
output_dir = dpr.get_folder_path_from_namespace(dpr.structure.models.bert)
model_folder = os.path.join(output_dir, "checkpoint-175")

explanations_json_path = os.path.join(
    os.getcwd(),
    'app',
    'static',
    'json',
    'lime_explanations.json'
)
with open(explanations_json_path, 'r', encoding='utf-8') as f:
    explanations_data = json.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128

tokenizer = AutoTokenizer.from_pretrained(model_folder)
model = AutoModelForSequenceClassification.from_pretrained(model_folder)
model.eval()

################################################################################

emails_path = os.path.join(os.getcwd(), 'app', 'static', 'csv', 'emails.csv')
emails_df = pd.read_csv(emails_path)

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
    subject = data.get("subject", "")
    body = data.get("body", "")
    combined_text = subject + " " + body

    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_label_idx = torch.max(probs, dim=-1)
    
    predicted_label = "Phishing" if predicted_label_idx.item() == 1 else "Legit"

    # LIME Explanation
    explainer = lime.lime_text.LimeTextExplainer(class_names=["Legit", "Phishing"])
    def predict_proba(texts):
        tokens = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**tokens).logits
        return torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

    exp = explainer.explain_instance(combined_text, predict_proba, num_features=10)
    lime_html = exp.as_html()

    return jsonify({
        "predicted_label": predicted_label,
        "confidence": confidence.item(),
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