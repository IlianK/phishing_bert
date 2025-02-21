import torch
import lime.lime_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Load model and tokenizer
def load_model(model_folder, device):
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    model = AutoModelForSequenceClassification.from_pretrained(model_folder)
    model.to(device)
    model.eval()
    return tokenizer, model


# Load explanations data from JSON file
def load_explanations(explanations_json_path):
    import json
    with open(explanations_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Predict phishing or legit
def predict_label(text, tokenizer, model, max_len=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_label_idx = torch.max(probs, dim=-1)
    
    predicted_label = "Phishing" if predicted_label_idx.item() == 1 else "Legit"
    return predicted_label, confidence.item()


# Generate LIME explanation
def explain_prediction(text, tokenizer, model, max_len=512):
    explainer = lime.lime_text.LimeTextExplainer(class_names=["Legit", "Phishing"])
    
    def predict_proba(texts):
        tokens = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
        with torch.no_grad():
            logits = model(**tokens).logits
        return torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    return exp.as_html()