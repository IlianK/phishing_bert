import pandas as pd
import os
import re
from langdetect import detect
import textstat
from tqdm import tqdm  # Import tqdm for progress bar

# Paths
project_root = os.getcwd()
print("Project Root:", project_root)

RAW_PATH = os.path.join(project_root, "data", "raw")
PREPROCESSED_PATH = os.path.join(project_root, "data", "preprocessed")
os.makedirs(PREPROCESSED_PATH, exist_ok=True)

# Load phishing terms
phishing_terms_file = os.path.join(RAW_PATH, "_phishing_terms.txt")
with open(phishing_terms_file, "r") as f:
    PHISHING_TERMS = [line.strip() for line in f.readlines() if line.strip()]
print(f"Loaded {len(PHISHING_TERMS)} phishing terms.")

# Load suspicious domains 
suspicious_domains_file = os.path.join(RAW_PATH, "_suspicious_domains.txt")
with open(suspicious_domains_file, "r") as f:
    SUSPICIOUS_DOMAINS = [line.strip() for line in f.readlines() if line.strip()]
print(f"Loaded {len(SUSPICIOUS_DOMAINS)} suspicious domains.")


# Function to count special characters
def count_special_chars(text):
    return len(re.findall(r"[!@#$%^&*()_+={}\[\]:;\"'<>,.?/\\|`~]", text))

# Function to count numbers
def count_numbers(text):
    return len(re.findall(r"\d", text))

# Function to count phishing terms
def count_phishing_terms(text, terms):
    count = 0
    for term in terms:
        count += len(re.findall(rf"\b{term}\b", text, re.IGNORECASE))
    return count

# Function to calculate average word length
def avg_word_length(text):
    words = str(text).split()
    return sum(len(word) for word in words) / len(words) if words else 0

# Function to count uppercase words
def count_uppercase_words(text):
    return len([word for word in str(text).split() if word.isupper()])

# Function to count URLs
def count_urls(text):
    return len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))

# Function to count HTML tags
def count_html_tags(text):
    return len(re.findall(r'<[^>]+>', text))

# Function to count exclamation marks
def count_exclamation_marks(text):
    return len(re.findall(r'!', text))

# Function to count question marks
def count_question_marks(text):
    return len(re.findall(r'\?', text))

# Function to count repeated characters
def count_repeated_chars(text):
    return len(re.findall(r'(.)\1{2,}', text))

# Function to count suspicious domains
def count_suspicious_domains(text, domains):
    return sum(text.lower().count(domain) for domain in domains)

# Function to count email addresses
def count_email_addresses(text):
    return len(re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text))

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


# Load raw email data
emails_path = os.path.join(RAW_PATH, "data_mail", "raw_combined.csv")
emails_df = pd.read_csv(emails_path)

# Fill NaN values in 'text' column
emails_df['text'] = emails_df['text'].fillna('')

# Extract features
emails_df['word_count'] = emails_df['text'].progress_apply(lambda x: len(str(x).split()))
emails_df['char_count'] = emails_df['text'].progress_apply(lambda x: len(str(x)))
emails_df['special_char_count'] = emails_df['text'].progress_apply(count_special_chars)
emails_df['number_count'] = emails_df['text'].progress_apply(count_numbers)
emails_df['phishing_term_count'] = emails_df['text'].progress_apply(lambda x: count_phishing_terms(x, PHISHING_TERMS))
emails_df['avg_word_length'] = emails_df['text'].progress_apply(avg_word_length)
emails_df['uppercase_word_count'] = emails_df['text'].progress_apply(count_uppercase_words)
emails_df['url_count'] = emails_df['text'].progress_apply(count_urls)
emails_df['html_tag_count'] = emails_df['text'].progress_apply(count_html_tags)
emails_df['repeated_char_count'] = emails_df['text'].progress_apply(count_repeated_chars)
emails_df['suspicious_domain_count'] = emails_df['text'].progress_apply(lambda x: count_suspicious_domains(x, SUSPICIOUS_DOMAINS))
emails_df['email_address_count'] = emails_df['text'].progress_apply(count_email_addresses)
emails_df['language'] = emails_df['text'].progress_apply(detect_language)
emails_df['readability_score'] = emails_df['text'].progress_apply(textstat.flesch_reading_ease)
emails_df['exclamation_count'] = emails_df['text'].progress_apply(count_exclamation_marks)
emails_df['question_count'] = emails_df['text'].progress_apply(count_question_marks)

# Select meta features
meta_features_df = emails_df[[ 
    'label', 'word_count', 'char_count', 'special_char_count', 
    'number_count', 'phishing_term_count', 'avg_word_length',
    'uppercase_word_count', 'url_count', 'html_tag_count',
    'repeated_char_count', 'suspicious_domain_count', 
    'email_address_count', 'language', 'readability_score',
    'exclamation_count', 'question_count'
]]

# Save the processed DataFrame
meta_features_path = os.path.join(PREPROCESSED_PATH, "meta.csv")
meta_features_df.to_csv(meta_features_path, index=False)
print(f"Meta features dataset saved to {meta_features_path}")