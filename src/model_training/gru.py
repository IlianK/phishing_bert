
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback

from safetensors.torch import save_file, load_file


##################################################################

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.helper_functions.path_resolver import DynamicPathResolver

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

##################################################################

# Setup and paths
def setup_paths():
    resolver = DynamicPathResolver(marker="README.md")
    structure = resolver.structure

    output_dir = resolver.get_folder_path_from_namespace(structure.models.gru)
    log_dir = os.path.join(output_dir, ".logs")

    return structure, output_dir, log_dir


# Load and preprocess data
def load_data(structure, data_amount):
    phishing_url_csv = structure.data.preprocessed.data_for_gru.phishing_urls_csv
    url_data = pd.read_csv(phishing_url_csv)
    le = LabelEncoder()
    url_data['status'] = le.fit_transform(url_data['status'])
    
    url_data = url_data.groupby('status', group_keys=False).apply(
        lambda x: x.sample(int(np.rint(len(x) / len(url_data) * data_amount)), random_state=42)
    ).reset_index(drop=True)
    
    print("\n=== Updated Class Distribution (Phishing = 0, Legit = 1)===")
    print(url_data['status'].value_counts(), "\n")

    return url_data, le


# Split dataset
def split_data(url_df):
    X_train, X_test, y_train, y_test = train_test_split(url_df['url'], url_df['status'], test_size=0.2, random_state=42, stratify=url_df['status'])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    print("\n=== Class Distribution After Splitting ===")
    print("TRAIN:\n", y_train.value_counts(), "\n")
    print("VALIDATION:\n", y_val.value_counts(), "\n")
    print("TEST:\n", y_test.value_counts(), "\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


# Tokenize URLs
def tokenize_urls(X_train, X_val, X_test):
    def tokenize_url(url, ngram=3):
        return [url[i:i+ngram] for i in range(len(url)-ngram+1)]
    
    def url_to_indices(tokens, vocab):
        return [vocab.get(ngram, 0) for ngram in tokens]

    X_train_tokens = X_train.apply(tokenize_url)
    X_val_tokens = X_val.apply(tokenize_url)
    X_test_tokens = X_test.apply(tokenize_url)

    all_ngrams = set(ngram for url in X_train_tokens for ngram in url)
    vocab = {ngram: idx for idx, ngram in enumerate(all_ngrams)}
    vocab_size = len(vocab)

    print(f"\n=== Vocabulary Information ===")
    print(f"Vocabulary size: {vocab_size}\n")

    X_train_indices = X_train_tokens.apply(lambda x: url_to_indices(x, vocab))
    X_val_indices = X_val_tokens.apply(lambda x: url_to_indices(x, vocab))
    X_test_indices = X_test_tokens.apply(lambda x: url_to_indices(x, vocab))

    return X_train_indices, X_val_indices, X_test_indices, vocab


# DataLoader
def prepare_dataloader(X_train_indices, X_val_indices, X_test_indices, y_train, y_val, y_test, max_len=100):
    train_dataset = URLDataset(X_train_indices, y_train, max_len=max_len)
    val_dataset = URLDataset(X_val_indices, y_val, max_len=max_len)
    test_dataset = URLDataset(X_test_indices, y_test, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("\n=== Dataset Sizes ===")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}\n")

    return train_loader, val_loader, test_loader


# URL Dataset class
class URLDataset(Dataset):
    def __init__(self, urls, labels, max_len=100):
        self.urls = urls
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url_indices = self.urls.iloc[idx]
        label = self.labels.iloc[idx]
        url_indices = url_indices[:self.max_len]  
        url_indices = url_indices + [0] * (self.max_len - len(url_indices)) 
        return {
            "input_ids": torch.tensor(url_indices, dtype=torch.long),
            "status": torch.tensor(label, dtype=torch.long),
        }


##################################################################


# GRU model class
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, max_len):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.attn_fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, labels=None):
        x = self.dropout(self.embedding(input_ids))  
        gru_out, _ = self.gru(x)  
        attn_weights = torch.tanh(self.attn_fc(gru_out)) 
        attn_scores = torch.sum(attn_weights * gru_out, dim=1) 
        logits = self.fc(attn_scores)

        if labels is not None: 
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}


# Train on epoch
def train_on_epoch(model, train_loader, optimizer, scheduler, device, writer, epoch, global_step, log_freq=100):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['status'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        logits = outputs['logits']

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        accuracy = correct / total

        # Log metrics
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/accuracy', accuracy, global_step)

        # Log learning rate 
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, global_step)

        # Print logs 
        if (batch_idx + 1) % log_freq == 0:
            print(f"Train Step [{global_step}] - "
                  f"Train Loss: {loss.item():.4f} - "
                  f"Train Accuracy: {accuracy:.4f} - "
                  f"Learning Rate: {current_lr:.6f}")

        # Track metrics 
        epoch_loss += loss.item()
        epoch_correct += correct
        epoch_total += total
        
        global_step += 1

    # Calculate and log 
    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_epoch_accuracy = epoch_correct / epoch_total
    writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
    writer.add_scalar('train/epoch_accuracy', avg_epoch_accuracy, epoch)

    print(f"\nEnd of Epoch {epoch + 1} - "
          f"Train Loss:     {avg_epoch_loss:.4f} - "
          f"Train Accuracy: {avg_epoch_accuracy:.4f}\n")

    return global_step


# Evaluate on epoch
def eval_on_epoch(model, val_loader, device, writer, epoch, global_step, log_freq=100):
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['status'].to(device)

            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

            total_loss += loss.item()

            # Log evaluation loss and accuracy
            eval_loss = total_loss / (batch_idx + 1)
            eval_accuracy = correct_preds / total_preds  
            writer.add_scalar('eval/loss', eval_loss, global_step)
            writer.add_scalar('eval/accuracy', eval_accuracy, global_step)

            # Print logs 
            if (batch_idx + 1) % log_freq == 0:
                print(f"Eval Step [{global_step}] - ",
                      f"Eval Loss: {eval_loss:.4f} - ", 
                      f"Eval Accuracy: {eval_accuracy:.4f}")

            global_step += 1
        
        avg_loss = total_loss / len(val_loader)
        avg_accuracy = correct_preds / total_preds
        writer.add_scalar('eval/epoch_loss', avg_loss, epoch)
        writer.add_scalar('eval/epoch_accuracy', avg_accuracy, epoch)

    print(f"\nEnd of Epoch {epoch + 1} - "
          f"Eval Loss:      {avg_loss:.4f} - "
          f"Eval Accuracy: {avg_accuracy:.4f}")

    return global_step


# Train loop
def train_evaluate(model, train_loader, val_loader, optimizer, scheduler, config, device, output_dir, log_dir):
    writer = SummaryWriter(log_dir)

    best_val_loss = float('inf')
    patience_counter = 0

    global_step = 1  

    for epoch in range(config['num_epochs']):
        print(f"\n--------------------------- Start Epoch {epoch + 1} ---------------------------")
        
        # Train phase
        global_step = train_on_epoch(
            model, train_loader, optimizer, scheduler, device, writer, epoch, global_step
        )

        # Eval phase
        global_step = eval_on_epoch(
            model, val_loader, device, writer, epoch, global_step
        )

        # Update learning rate
        if scheduler:
            scheduler.step(global_step)

        # Save model
        torch.save(model.state_dict(), os.path.join(output_dir, "gru_model.pth"))

        # Save best model
        if global_step < best_val_loss:
            best_val_loss = global_step
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "gru_model_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    writer.close()
    return


# Evaluate on test data
def evaluate_on_test(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            labels = batch['status']

            outputs = model(input_ids)
            logits = outputs['logits']

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# Inference on URLs
def inference_on_urls(model, urls, vocab, device, max_len=100):
    model.eval()
    all_preds = []
    all_probs = []

    def tokenize_url(url, ngram=3):
        return [url[i:i+ngram] for i in range(len(url)-ngram+1)]

    def url_to_indices(tokens, vocab):
        return [vocab.get(ngram, 0) for ngram in tokens]

    tokenized_urls = [tokenize_url(url) for url in urls]
    url_indices = [url_to_indices(tokens, vocab) for tokens in tokenized_urls]

    url_indices = [indices[:max_len] + [0] * (max_len - len(indices)) for indices in url_indices]
    input_ids = torch.tensor(url_indices, dtype=torch.long).to(device)

    with torch.no_grad():
        for input_id in input_ids:
            input_id = input_id.unsqueeze(0)  
            outputs = model(input_id)  
            logits = outputs['logits'] 
            
            probs = torch.softmax(logits, dim=1).cpu().numpy() 
            pred = torch.argmax(logits, dim=1).item() 
            
            all_preds.append(pred)
            all_probs.append(probs[0])  

    return all_preds, all_probs


# Load trained model
def load_model(output_dir, vocab_size, config):
    model = GRUModel(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        max_len=config['max_len']
    )
    
    model.load_state_dict(torch.load(os.path.join(output_dir, "gru_model.pth")))
    model.eval()
    return model