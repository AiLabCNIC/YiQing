import re


def clean_text(text):
    text = re.sub(r'https?://\S+', '', text)  # Remove link
    text = re.sub(r'\n', ' ', text)  # Remove line breaks
    text = re.sub('\s+', ' ', text).strip()  # Remove leading, trailing, and extra spacess
    return text


def process_text(df):
    df['text'] = df['text'].apply(lambda x: clean_text(x))
    return df
