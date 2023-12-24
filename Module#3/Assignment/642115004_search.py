import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import string
import argparse

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words and token not in string.punctuation]
    return ' '.join(tokens)

def preprocess_bigrams(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words and token not in string.punctuation]
    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return ' '.join(bigrams)

def tfidf_search(keyword, descriptions):
    vectorizer = TfidfVectorizer(preprocessor=preprocess_bigrams)
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    keyword_vector = vectorizer.transform([preprocess_bigrams(keyword)])

    scores = np.dot(tfidf_matrix, keyword_vector.T).toarray().flatten()
    sorted_indices = np.argsort(scores)[::-1]

    results = []
    for idx in sorted_indices[:5]:
        if scores[idx] > 0:
            results.append((scores[idx], descriptions.iloc[idx].strip()))

    return results

def bm25_search(keyword, descriptions):
    tokenized_descriptions = [word_tokenize(desc.lower()) for desc in descriptions]
    bm25 = BM25Okapi(tokenized_descriptions)
    scores = bm25.get_scores(word_tokenize(keyword.lower()))

    sorted_indices = np.argsort(scores)[::-1]

    results = []
    for idx in sorted_indices[:5]:
        if scores[idx] > 0:
            results.append((scores[idx], descriptions.iloc[idx].strip()))

    return results

def get_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    description = data['job_description']
    cleaned_description = description.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    cleaned_description = cleaned_description.apply(lambda s: s.lower())
    cleaned_description = cleaned_description.apply(lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), '')))
    cleaned_description = cleaned_description.drop_duplicates()
    return cleaned_description

   

def search_and_display_results(keyword, data_file_path, method):
    cleaned_description = get_and_clean_data(data_file_path)
    
    if method == 'tfidf':
        results = tfidf_search(keyword, cleaned_description)
    elif method == 'bm25':
        results = bm25_search(keyword, cleaned_description)
    else:
        raise ValueError("Invalid search method. Choose 'tfidf' or 'bm25'.")

    print(f"\nTop 5 matching jobs ({method.upper()}): {keyword}")
    for score, result in results:
        print(f"Score: {score:.4f}, {result}")
        print("\n================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Job Search CLI")
    parser.add_argument("keyword", type=str, help="Search keyword")
    parser.add_argument("data_file_path", type=str, help="Path to the data file")
    parser.add_argument("--method", type=str, choices=['tfidf', 'bm25'], default='tfidf',
                        help="Search method (tfidf or bm25)")

    args = parser.parse_args()
    search_and_display_results(args.keyword, args.data_file_path, args.method)
    