import re
import pandas as pd


# Read dataset
def read_dataset(file_path):
    return pd.read_csv(file_path)


# Process missing values
def process_text_columns(df):
    df['subject'] = df['subject'].fillna('[NO_SUBJECT]').astype(str)
    df['body'] = df['body'].fillna('[NO_BODY]').astype(str)
    return df


# Clean text 
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    substitutions = [
        (r'https?://\S+|www\.\S+', '[URL]'),                                    # Replace URLs
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]'),     # Replace emails
        (r'-{2,}', ' '), (r'!{2,}', '!'), (r'\?{2,}', '?'),                     # Remove repeated punctuation
        (r'[_+*]{2,}', ' '), (r'[=+]{3,}', ' '), (r'[~]{3,}', ' '),
        (r'[#]{3,}', ' '), (r'[<]{3,}', ' '), (r'[>]{3,}', ' ')
    ]
    for pattern, repl in substitutions:
        text = re.sub(pattern, repl, text)
    return text.strip()


# Combine subject and body 
def combine_text_fields(df):
    df['subject'] = df['subject'].apply(clean_text)
    df['body'] = df['body'].apply(clean_text)
    df['text'] = df['subject'] + " [SEP] " + df['body']
    return df


# Full preprocessing
def prepare_bert_data(df):
    df = process_text_columns(df)
    df = combine_text_fields(df)
    return df[['text', 'label']]


