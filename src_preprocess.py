import pandas as pd
import string
import re

def clean_text(text):
    # Remove punctuation and lowercase everything
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df['text'] = df['text'].apply(clean_text)
    return df