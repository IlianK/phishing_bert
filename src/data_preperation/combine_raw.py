import os
import pandas as pd

project_root = os.getcwd()
print("Project Root:", project_root)
RAW_PATH = os.path.join(project_root, "data", "raw", "data_mail")

combined_data = []

# Read all csv files
for csv_file in os.listdir(RAW_PATH):
    if csv_file.endswith(".csv"):
        file_path = os.path.join(RAW_PATH, csv_file)
        df = pd.read_csv(file_path)
        
        if 'label' in df.columns and 'subject' in df.columns and 'body' in df.columns:
            df['text'] = df['subject'] + ' ' + df['body']
            combined_data.append(df[['label', 'text']])

# Concatenate and save
combined_df = pd.concat(combined_data, ignore_index=True)
combined_csv_path = os.path.join(project_root, "data", "raw", "combined_raw.csv")
combined_df.to_csv(combined_csv_path, index=False)

print(f"Combined data saved to: {combined_csv_path}")
