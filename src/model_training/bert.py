import os
import pandas as pd
import torch, evaluate
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import sys
from transformers import TrainerCallback
from copy import deepcopy
from tqdm import tqdm 
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

##################################################################

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.helper_functions.path_resolver import DynamicPathResolver
from src.helper_functions.visualization import plot_roc_curve, plot_precision_recall, plot_confusion_matrix

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

##################################################################

# Setup and paths
def setup_paths():
    resolver = DynamicPathResolver(marker="README.md")
    structure = resolver.structure

    output_dir = resolver.get_folder_path_from_namespace(structure.models.bert)
    log_dir = os.path.join(output_dir, ".logs")
    
    return structure, output_dir, log_dir


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data  # pandas DataFrame with 'text' and 'label'
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the text and label
        text = str(self.data.iloc[idx]["text"])
        label = self.data.iloc[idx]["label"]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',  # Or 'longest' if you prefer dynamic padding
            max_length=self.max_len,
            return_tensors="pt"
        )

        # We squeeze to remove the unnecessary extra dimensions
        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # (1, max_len) -> (max_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (1, max_len) -> (max_len,)
            "labels": torch.tensor(label, dtype=torch.long)
        }


# Load and preprocess data
def load_data(structure, data_amount):
    mail_csv = structure.data.preprocessed.data_for_bert.combined_for_bert_csv
    mail_data = pd.read_csv(mail_csv)
    
    mail_data.dropna(subset=["text", "label"], inplace=True)
    mail_data['label'] = mail_data['label'].astype(int)
    mail_data = mail_data.groupby('label', group_keys=False).apply(
        lambda x: x.sample(
            int(np.rint(len(x) / len(mail_data) * data_amount)), 
            random_state=42
        )
    ).reset_index(drop=True)
    
    print("\n=== Updated Class Distribution ===")
    print(mail_data['label'].value_counts(), "\n")

    return mail_data


# Split dataset
def split_data(data, test_size=0.2, eval_size=0.2):
    temp_data, test_data = train_test_split(
        data,
        test_size=test_size,  
        stratify=data['label'],
        random_state=42
    )
    train_data, eval_data = train_test_split(
        temp_data,
        test_size=eval_size,
        stratify=temp_data['label'], 
        random_state=42
    )

    print(f"Training data size: {len(train_data)}")
    print(f"Evaluation data size: {len(eval_data)}")
    print(f"Test data size: {len(test_data)}")

    return train_data, eval_data, test_data


# Preprocess and create dataloaders
def create_custom_datasets(train_data, eval_data, test_data, tokenizer, max_len=128):
    train_dataset = CustomDataset(train_data, tokenizer, max_len)
    eval_dataset = CustomDataset(eval_data, tokenizer, max_len)
    test_dataset = CustomDataset(test_data, tokenizer, max_len)

    return train_dataset, eval_dataset, test_dataset


# Create model and tokenizer
def create_model_and_tokenizer(bert_type, special_tokens, device):
    id2label = {0: "legit", 1: "phishing"}
    label2id = {"legit": 0, "phishing": 1}

    tokenizer = AutoTokenizer.from_pretrained(bert_type)
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_type, 
        num_labels=2,
        id2label=id2label,
        label2id=label2id
        ).to(device)

    if special_tokens:
        tokenizer.add_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))

    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    return model, tokenizer


# Compute metrics 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

# CustomCallback with logging
class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


# Trainer
def train_evaluate(model, tokenizer, train_dataset, eval_dataset, config, data_collator, output_dir, log_dir):

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=config['evaluation_strategy'],
        save_strategy=config['save_strategy'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        logging_steps=config['logging_steps'],
        load_best_model_at_end=True,
        save_total_limit=2,
        logging_dir=log_dir,
        warmup_steps=config['warmup_steps'],
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(CustomCallback(trainer))
    trainer.train()
    trainer.evaluate()

    return trainer


# Evaluate on test set
def evaluate_on_test(model, test_data, data_collator, batch_size, device):
    test_loader = DataLoader(
        test_data, 
        batch_size, 
        collate_fn=data_collator
        )

    model.eval()  
    true_labels = []
    predicted_labels = []
    probs = []

    with torch.no_grad(): 
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())
            probs.extend(probabilities.cpu().numpy())

    return true_labels, predicted_labels, probs


def inference(model, texts, true_labels, tokenizer, max_length, device):
    encoded_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)

    model.eval()
    predicted_labels = []
    probabilities = []

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        predicted_labels.extend(preds.cpu().numpy())
        probabilities.extend(probs.cpu().numpy())

    return true_labels, predicted_labels, probabilities


# Load model
def load_model_from_checkpoint(output_dir, checkpoint_num, device):
    checkpoint_folder = f"checkpoint-{checkpoint_num}"
    checkpoint_path = os.path.join(output_dir, checkpoint_folder)
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)

    print(f"Model loaded from {checkpoint_path} onto {device}.")
    
    return model, tokenizer
