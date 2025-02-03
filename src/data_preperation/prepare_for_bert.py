import os
import pandas as pd
import re

project_root = os.getcwd()
print("Project Root:", project_root)

RAW_PATH = os.path.join(project_root, "data", "raw", "data_mail")
PREPROCESSED_PATH = os.path.join(project_root, "data", "preprocessed", "data_for_bert")
os.makedirs(PREPROCESSED_PATH, exist_ok=True)


def read_csv_files(directory_path):
    dataframes = []
    print(f"Reading CSV files from {directory_path}\n")

    for file in os.listdir(directory_path):
        if file.endswith('.csv'):
            file_path = os.path.join(directory_path, file)
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')

                if 'label' in df.columns:
                    df = df.dropna(subset=['label'])  
                    df['label'] = df['label'].astype(int) 

                dataframes.append(df)
                print(f"Loaded {file}: {len(df)} rows")

                if 'label' in df.columns:
                    class_counts = df['label'].value_counts()
                    total = len(df)
                    print("Class Balance:")
                    for cls, count in class_counts.items():
                        print(f"  {cls}: {count} ({count / total * 100:.2f}%)")
                    print()
                else:
                    print("  Warning: 'label' column not found in this file.\n")
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return dataframes


def preprocess_and_clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'http[s]?://\S+', '[URL]', text)
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '[DATE]', text)
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}\b(?!\s*(inches|inch|ft|feet|cm|mm|m))', '[DATE]', text)
    text = re.sub(r'\b\d{1,2}:\d{2}\s*[apAP][mM]?\b', '[TIME]', text)
    text = re.sub(r'\b\d{1,2}:\d{2}\b', '[TIME]', text)
    text = re.sub(r'[^\w\s.,!?[\]()]', ' ', text)
    text = re.sub(r'\s*\n+\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('""', '')
    return text.strip()


def prepare_data(dataframes):
    combined_df = pd.concat(dataframes, ignore_index=True)
    if 'label' not in combined_df.columns or 'subject' not in combined_df.columns or 'body' not in combined_df.columns:
        raise ValueError("One or more required columns ('label', 'subject', 'body') are missing in the combined data.")
    
    combined_df = combined_df.dropna(subset=['label']) 
    combined_df['label'] = combined_df['label'].astype(int)  
    combined_df['subject'] = combined_df['subject'].apply(preprocess_and_clean_text)
    combined_df['body'] = combined_df['body'].apply(preprocess_and_clean_text)
    combined_df['text'] = combined_df['subject'] + " [SEP] " + combined_df['body']
    combined_df = combined_df[['label', 'text']]
    
    class_counts = combined_df['label'].value_counts()
    total = len(combined_df)
    print("Combined Class Balance:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} ({count / total * 100:.2f}%)")

    return combined_df


if __name__ == "__main__":
    dataframes = read_csv_files(RAW_PATH)
    processed_data = prepare_data(dataframes)

    save_path = os.path.join(PREPROCESSED_PATH, "combined_for_bert.csv")
    processed_data.to_csv(save_path, index=False)
    print(f"Processed data saved to {save_path}")
