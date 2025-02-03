import os
import pandas as pd

project_root = os.getcwd()
print("Project Root:", project_root)

PHISHING_URL_PATH = os.path.join(project_root, "data", "preprocessed", "data_for_gru", "phishing_urls.csv")
AUGMENTED_PHISHING_URL_PATH = os.path.join(project_root, "data", "preprocessed", "data_for_gru", "phishing_urls_augmented.csv")
OUTPUT_PATH = os.path.join(project_root, "data", "preprocessed", "data_for_gru", "phishing_urls_combined.csv")

def concat_csv(original_csv_path, augmented_csv_path, output_csv_path):
    phishing_data = pd.read_csv(original_csv_path)
    augmented_phishing_data = pd.read_csv(augmented_csv_path)
    
    combined_data = pd.concat([phishing_data, augmented_phishing_data], ignore_index=True)
    combined_data = combined_data.drop_duplicates(subset=['url'], keep='first')
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    combined_data.to_csv(output_csv_path, index=False)
    
    class_balance = combined_data['status'].value_counts()
    class_percentage = combined_data['status'].value_counts(normalize=True) * 100
    
    print(f"Class balance (count and percentage):\n")
    for status, count in class_balance.items():
        percentage = class_percentage[status]
        print(f"Class {status}: {count} instances ({percentage:.2f}%)")

if __name__ == "__main__":
    concat_csv(PHISHING_URL_PATH, AUGMENTED_PHISHING_URL_PATH, OUTPUT_PATH)
