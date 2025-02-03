import os
import random
import pandas as pd
import re
from urllib.parse import urlparse

project_root = os.getcwd()
print("Project Root:", project_root)

ALEXA_PATH = os.path.join(project_root, "data", "raw", "data_url", "alexa_top_million.csv")
PREPROCESSED_PATH = os.path.join(project_root, "data", "preprocessed", "data_for_gru", "phishing_urls_augmented.csv")

LOOKALIKE_REPLACEMENTS = {
    'o': '0',
    '0': 'o',
    'a': '4',
    '4': 'a',
    'l': '1',
    '1': 'l',
    'I': '1',
    'L': '1',
    'E': '3',
    '3': 'E',
    'i': 'l',
    'l': 'i',
    'O': '0',
    'S': '$',
    '$': 'S',
    'Z': '2',
    '2': 'Z',
    'B': '8',
    '8': 'B',
    'G': '6',
    '6': 'G',
    'T': '7',
    '7': 'T',
    'M': 'W',
    'W': 'M',
    'P': 'R',
    'R': 'P',
    'Y': 'V',
    'V': 'Y'
}

KEYWORDS = [
    'secure-login', 'login-secure', 'identify', 'auth', 'signin', 'signup', 'account', 'user', 'myaccount', 'acc',
    'verify', 'payment', 'support', 'service', 'update', 'account-info', 'bank', 'secure', 'check', 'help', 'customer',
    'login-help', 'password', 'security', 'authentication', 'confirm', 'validate', 'confirm-account', 'user-login',
    'myaccount', 'member', 'login-verify', 'youraccount', 'access', 'account-verify', 'support-login', 'phishing', 'alert',
    'notifications', 'services', 'login-info', 'secure-account', 'safe', 'portal', 'alerts', 'messages', 'settings', 'service-update',
    'update-password', 'contact', 'claim', 'billing', 'billing-info', 'pay-now', 'reset-password', 'recover', 'admin', 'admin-login',
    'checkout', 'payment-update', 'security-check', 'locked', 'action-required', 'confirm-login', 'subscription', 'notification', 
    'download', 'secure-checkout', 'transaction', 'transaction-confirm', 'payment-secure', 'info-update', 'login-alert', 'transaction-alert',
    'payment-confirm', 'my-secure', 'account-update', 'change-password', 'security-alert', 'secure-portal', 'secure-login-info', 
    'payment-info', 'alerts-update', 'user-verify', 'action-needed', 'password-reset', 'secure-user', 'transaction-update', 'login-confirm'
]


def extract_tld_from_url(url):
    regex = r'^(?:https?://)?([^/]+)'  
    match = re.match(regex, url)
    if match:
        domain_tld = match.group(1)
        parts = domain_tld.rsplit('.', 1)
        if len(parts) == 2:
            domain, tld = parts
            return domain, tld
    return url, ''


def replace_random_lookalike_characters_in_domain(domain, num_replacements=1):
    modified_domain = list(domain)
    available_chars = list(LOOKALIKE_REPLACEMENTS.keys())
    chars_to_replace = [char for char in available_chars if char in domain]
    
    if not chars_to_replace:
        return insert_random_character_in_domain(domain)  
    
    num_replacements = min(num_replacements, len(chars_to_replace))
    chars_to_replace = random.sample(chars_to_replace, num_replacements)
    
    for char in chars_to_replace:
        modified_domain = [LOOKALIKE_REPLACEMENTS.get(c, c) if c == char else c for c in modified_domain]
    
    modified_url = ''.join(modified_domain)
    return modified_url


def insert_random_character_in_domain(domain):
    insert_position = random.randint(0, len(domain))
    random_char = random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    modified_domain = domain[:insert_position] + random_char + domain[insert_position:]
    return modified_domain


def remove_random_character_in_domain(domain):
    if len(domain) > 1:
        remove_position = random.randint(0, len(domain) - 1)
        modified_domain = domain[:remove_position] + domain[remove_position + 1:]
    else:
        modified_domain = insert_random_character_in_domain(domain)
    return modified_domain


def add_https_to_legit_urls(url):
    if not url.startswith(('http://', 'https://')) and random.random() < 0.5:
        url = 'https://' + url
    return url


def add_keyword_as_subdomain(domain):
    keyword = random.choice(KEYWORDS)
    modified_url = keyword + '.' + domain
    return modified_url


def add_keyword_with_hyphen(domain):
    keyword = random.choice(KEYWORDS)
    modified_url = keyword + '-' + domain
    return modified_url


def augment_url(url, is_legit=False):
    augmented = ''
    if is_legit:
        return add_https_to_legit_urls(url)
    
    rand_num = random.random()
    if rand_num < 0.4:
        augmented = replace_random_lookalike_characters_in_domain(url, num_replacements=1)
    elif rand_num < 0.7:
        augmented = insert_random_character_in_domain(url)
    elif rand_num < 0.85:
        augmented = add_keyword_as_subdomain(url)
    else:
        augmented = add_keyword_with_hyphen(url)

    protocol_rand = random.random()
    if protocol_rand < 0.66:
        augmented = 'http://' + augmented
    elif protocol_rand < 0.7:
        augmented = 'https://' + augmented
    else:
        augmented = augmented
    return augmented


def augment_csv(input_path, output_path):
    PHISHING_TLDS = ['.cc', '.xyz', '.top', '.club', '.loan', '.online', '.site', '.tech', '.work', '.win']
    
    alexa_data = pd.read_csv(input_path, header=None, names=['idx', 'url'], dtype={'idx': str, 'url': str})
    augmented_phishing = []
    #alexa_data = alexa_data.head(50)
    
    for _, row in alexa_data.iterrows():
        original_url = row['url']
        domain, tld = extract_tld_from_url(original_url)

        # Legit augment
        augmented_legit_url = augment_url(domain, is_legit=True)
        augmented_legit_url = augmented_legit_url + '.' + tld if tld else original_url
        augmented_phishing.append({'url': augmented_legit_url, 'status': 1})        
        
        # Phishing augment
        augmented_phishing_url = augment_url(domain, is_legit=False)
        if random.random() < 0.5:
            phishing_tld = random.choice(PHISHING_TLDS)
        phishing_tld = ''
        augmented_phishing_url = augmented_phishing_url + '.' + (phishing_tld if phishing_tld != '' else tld)
        augmented_phishing.append({'url': augmented_phishing_url, 'status': 0})

    augmented_phishing_df = pd.DataFrame(augmented_phishing)
    augmented_phishing_df.to_csv(output_path, index=False)
    print(f"Combined and augmented dataset saved to {output_path}")


if __name__ == "__main__":
    augment_csv(ALEXA_PATH, PREPROCESSED_PATH)
