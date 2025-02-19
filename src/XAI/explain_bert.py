import lime
import lime.lime_text
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import sys, os
sys.path.append(os.path.abspath('../../src'))
from helper_functions.path_resolver import DynamicPathResolver

dpr = DynamicPathResolver(marker="README.md")
paths = dpr.structure

models_folder = dpr.get_folder_path_from_namespace(paths.models.bert)
tokenizer = AutoTokenizer.from_pretrained(models_folder)
model = AutoModelForSequenceClassification.from_pretrained(models_folder)
model.eval()


def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
    return probs


def explain_text(text):
    explainer = lime.lime_text.LimeTextExplainer(class_names=["Legit", "Phishing"])
    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    return exp.as_html()
