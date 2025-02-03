import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator 
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc, confusion_matrix
import seaborn as sns
import os

from wordcloud import WordCloud 


########################## EVENT PLOTS ##########################

def extract_metrics_from_events(log_dir):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()

    # Extract train metrics
    train_loss_steps = [x.step for x in event_acc.Scalars('train/loss')]
    train_loss_values = [x.value for x in event_acc.Scalars('train/loss')]
    train_accuracy_steps = [x.step for x in event_acc.Scalars('train/accuracy')]
    train_accuracy_values = [x.value for x in event_acc.Scalars('train/accuracy')]

    # Extract eval metrics
    eval_loss_steps = [x.step for x in event_acc.Scalars('eval/loss')]
    eval_loss_values = [x.value for x in event_acc.Scalars('eval/loss')]
    eval_accuracy_steps = [x.step for x in event_acc.Scalars('eval/accuracy')]
    eval_accuracy_values = [x.value for x in event_acc.Scalars('eval/accuracy')]

    return {
        'train_loss_steps': train_loss_steps,
        'train_loss_values': train_loss_values,
        'train_accuracy_steps': train_accuracy_steps,
        'train_accuracy_values': train_accuracy_values,
        'eval_loss_steps': eval_loss_steps,
        'eval_loss_values': eval_loss_values,
        'eval_accuracy_steps': eval_accuracy_steps,
        'eval_accuracy_values': eval_accuracy_values
    }


# Extract all metrics from events
def extract_all_metrics_from_events(log_dir):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()

    scalar_tags = event_acc.Tags()['scalars']
    metrics = {}

    for tag in scalar_tags:
        category, metric_type = tag.split('/', 1)

        if category == 'train' or category == 'eval':
            steps = [x.step for x in event_acc.Scalars(tag)]
            values = [x.value for x in event_acc.Scalars(tag)]

            if category not in metrics:
                metrics[category] = {}

            metrics[category][metric_type] = {
                'steps': steps,
                'values': values
            }

    print("Existing Metrics Key Paths:")
    def print_key_paths(metrics_dict, path=""):
        for key in metrics_dict:
            if isinstance(metrics_dict[key], dict):
                print_key_paths(metrics_dict[key], path + key + " - ")
            else:
                print(path + key)

    print_key_paths(metrics)
    return metrics


# Extract loss accuracy from events
def extract_loss_and_accuracy_metrics(metrics):
    train_loss_steps = metrics['train'].get('loss', {}).get('steps', [])
    eval_loss_steps = metrics['eval'].get('loss', {}).get('steps', [])
    train_loss_values = metrics['train'].get('loss', {}).get('values', [])
    eval_loss_values = metrics['eval'].get('loss', {}).get('values', [])

    train_accuracy_steps = metrics['train'].get('train_accuracy', {}).get('steps', [])
    eval_accuracy_steps = metrics['eval'].get('accuracy', {}).get('steps', [])
    train_accuracy_values = metrics['train'].get('train_accuracy', {}).get('values', [])
    eval_accuracy_values = metrics['eval'].get('accuracy', {}).get('values', [])

    return {
        "train_loss_steps": train_loss_steps,
        "eval_loss_steps": eval_loss_steps,
        "train_loss_values": train_loss_values,
        "eval_loss_values": eval_loss_values,
        "train_accuracy_steps": train_accuracy_steps,
        "eval_accuracy_steps": eval_accuracy_steps,
        "train_accuracy_values": train_accuracy_values,
        "eval_accuracy_values": eval_accuracy_values
    }


def list_scalar_tags_and_histograms(log_dir):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()

    scalar_tags = event_acc.Tags()['scalars']
    histogram_tags = event_acc.Tags()['histograms']

    print("Scalar Tags:")
    for scalar_tag in scalar_tags:
        print(scalar_tag)

    print("\nHistogram Tags:")
    for histogram_tag in histogram_tags:
        print(histogram_tag)
        

def plot_scalar_metric(log_dir, scalar_tag):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    scalar_values = [x.value for x in event_acc.Scalars(scalar_tag)]
    steps = [x.step for x in event_acc.Scalars(scalar_tag)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, scalar_values, label=scalar_tag)
    plt.xlabel('Steps')
    plt.ylabel(scalar_tag)
    plt.title(f'{scalar_tag} vs. Steps')
    plt.legend()
    plt.grid(True)
    plt.show()


########################## EXPLORE PLOTS ##########################

def plot_text_length_distribution_with_ranges(dfs, file_names=None):
    if isinstance(dfs, list):
        num_datasets = len(dfs)
    else:
        dfs = [dfs]  
        num_datasets = 1
        file_names = ["Dataset"]  
    
    num_cols = 3  
    num_rows = math.ceil(num_datasets / num_cols)  
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 6 * num_rows), sharey=True)
    
    bins = [0, 50, 100, 200, 300, 400, 512, 1024, 2048]  #
    bin_labels = ['0-50', '51-100', '101-200', '201-300', '301-400', '401-512', '513-1024', '>1024']
    axes = axes.flatten()
    
    for i, (df, file_name) in enumerate(zip(dfs, file_names)):
        df['text_length'] = df['text'].apply(lambda x: len(x.split()))
        df['length_range'] = pd.cut(df['text_length'], bins=bins, labels=bin_labels, right=False)
        length_distribution = df['length_range'].value_counts().sort_index()
        
        sns.barplot(
            x=length_distribution.index,
            y=length_distribution.values,
            ax=axes[i],
            palette='RdYlBu',
            hue=length_distribution.index,
        )
        axes[i].set_title(f'Text Length Ranges - {file_name}')
        axes[i].set_xlabel('Length Range')
        axes[i].set_ylabel('Frequency' if i == 0 else '')
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_wordclouds(dfs, file_names):
    num_datasets = len(dfs)
    fig, axes = plt.subplots(2, num_datasets, figsize=(5 * num_datasets, 10))
    
    for i, (df, file_name) in enumerate(zip(dfs, file_names)):
        for label in [0, 1]:
            if label in df['label'].unique():
                text = ' '.join(df[df['label'] == label]['text'])
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                row = 0 if label == 0 else 1
                axes[row, i].imshow(wordcloud, interpolation='bilinear')
                axes[row, i].axis('off')
                label_name = 'Non-Phishing' if label == 0 else 'Phishing'
                axes[row, i].set_title(f'{label_name} - {file_name}')
    
    plt.tight_layout()
    plt.show()


def generate_wordcloud(df, column, label, language, title):
    text = ' '.join(df[df['label'] == label][column])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{title} for {language}')
    plt.show()


########################## INFERENCE ##########################

def display_inference_results(texts, true_labels, predicted_labels, probabilities, class_names=None):
    if class_names is None:
        class_names = ["Class 0", "Class 1"]

    print("\n---- Model Predictions ----")
    for text, true_label, pred, prob in zip(texts, true_labels, predicted_labels, probabilities):
        prob_str = ", ".join([f"{class_names[i]}: {p:.2f}" for i, p in enumerate(prob)])
        print(f"\nText: {text}")
        print(f"True Label: {class_names[true_label]}\nPred Label: {class_names[pred]}")
        print(f"     Probs: {prob_str}\n")
        print("-" * 60)


########################## TRAINING PLOTS ##########################

def plot_loss_accuracy(extracted_metrics, output_dir):
    train_loss_steps = extracted_metrics.get("train_loss_steps", [])
    eval_loss_steps = extracted_metrics.get("eval_loss_steps", [])
    train_loss_values = extracted_metrics.get("train_loss_values", [])
    eval_loss_values = extracted_metrics.get("eval_loss_values", [])

    train_accuracy_steps = extracted_metrics.get("train_accuracy_steps", [])
    eval_accuracy_steps = extracted_metrics.get("eval_accuracy_steps", [])
    train_accuracy_values = extracted_metrics.get("train_accuracy_values", [])
    eval_accuracy_values = extracted_metrics.get("eval_accuracy_values", [])

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_steps, train_loss_values, label="Training Loss")
    plt.plot(eval_loss_steps, eval_loss_values, label="Evaluation Loss", linestyle='--')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/loss_plot.png")
    plt.show()

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy_steps, train_accuracy_values, label="Training Accuracy")
    plt.plot(eval_accuracy_steps, eval_accuracy_values, label="Evaluation Accuracy", linestyle='--')
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training and Evaluation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/accuracy_plot.png")
    plt.show()


def plot_loss_and_accuracy(metrics, output_dir):
    train_accuracy_steps = metrics.get("train_accuracy_steps", [])
    eval_accuracy_steps = metrics.get("eval_accuracy_steps", [])
    train_loss_values = metrics.get("train_loss_values", [])
    eval_loss_values = metrics.get("eval_loss_values", [])
    train_accuracy_values = metrics.get("train_accuracy_values", [])
    eval_accuracy_values = metrics.get("eval_accuracy_values", [])

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy_steps, train_loss_values, label="Training Loss")
    plt.plot(eval_accuracy_steps, eval_loss_values, label="Evaluation Loss", linestyle='--')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/loss_plot.png")
    plt.show()

    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy_steps, train_accuracy_values, label="Training Accuracy")
    plt.plot(eval_accuracy_steps, eval_accuracy_values, label="Evaluation Accuracy", linestyle='--')
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training and Evaluation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/accuracy_plot.png")
    plt.show()


def plot_roc_curve(labels, probs, output_dir):
    probs = np.array(probs)
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])  
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.show()


def plot_precision_recall(labels, probs, output_dir):
    probs = np.array(probs)
    precision, recall, _ = precision_recall_curve(labels, probs[:, 1]) 

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.show()


def plot_confusion_matrix(cm, output_dir, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{title}.png"))
    plt.show()


def display_classification_report(true_labels, predicted_labels, target_names=None):
    report = classification_report(true_labels, predicted_labels, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df

########################## DATA EXPLORATION ##########################

def plot_countplot(df, column, title, x_labels=None, rotation=0):
    sns.countplot(x=column, data=df)
    plt.title(title)
    if x_labels:
        plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=rotation)
    plt.show()


def plot_bar(data, title, xlabel_rotation=0, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    sns.barplot(x=data.index, y=data.values)
    plt.xticks(rotation=xlabel_rotation)
    plt.title(title)
    plt.show()


def plot_pie(labels, values, title, colors, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title(title)
    plt.ylabel('')
    plt.show()


def plot_boxplot(df, x, y, title):
    sns.boxplot(x=x, y=y, data=df)
    plt.title(title)
    plt.show()


def plot_correlation_heatmap(df, numeric_cols, title, figsize=(10, 6)):
    correlation_matrix = df[numeric_cols].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(title)
    plt.show()


def plot_histogram(series, title, bins=10):
    plt.figure(figsize=(10, 6))
    plt.hist(series, bins=bins, color='lightcoral', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()
    

def encode_categorical_columns(df, columns):
    for column in columns:
        df[column] = df[column].astype(int)
    return df