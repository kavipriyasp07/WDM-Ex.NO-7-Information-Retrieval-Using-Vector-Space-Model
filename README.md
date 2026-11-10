# WDM-Ex.NO-7-Information-Retrieval-Using-Vector-Space-Model
**Information Retrieval Using Vector Space Model in Python**<br>
## Name: Kavipriya SP
## Date: 
**AIM:**<br>
To implement Information Retrieval Using Vector Space Model in Python.<br>
**Description:**<br>
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and sklearn to demonstrate Information Retrieval using the Vector Space Model.<br>
**Procedure:**
Define sample documents.
Preprocess text data by tokenizing, removing stopwords, and punctuation.
Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
Execute a sample query and display the search results along with similarity scores.
**Program:**
```
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

# Download required NLTK packages
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Documents
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?",
}

# Text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())

    tokens = [
        token for token in tokens
        if token not in stopwords.words("english")
        and token not in string.punctuation
    ]

    return " ".join(tokens)


# Preprocess documents
preprocessed_docs = {
    doc_id: preprocess_text(doc)
    for doc_id, doc in documents.items()
}

print("Preprocessed Documents:")
print(preprocessed_docs)

# TF-IDF calculation
tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(
    preprocessed_docs.values()
)


# Search function
def search(query, tfidf_matrix, tfidf_vectorizer):

    query = input("Enter your query: ")

    preprocessed_query = preprocess_text(query)

    query_vector = tfidf_vectorizer.transform(
        [preprocessed_query]
    )

    similarity_scores = cosine_similarity(
        query_vector, tfidf_matrix
    ).flatten()

    search_results = []

    for i, score in enumerate(similarity_scores):
        search_results.append((
            list(documents.keys())[i],
            documents[list(documents.keys())[i]],
            score
        ))

    search_results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    return search_results


# Execute search
query = input("Enter your query: ")

# Search function without another input prompt
def search(query, tfidf_matrix, tfidf_vectorizer):

    preprocessed_query = preprocess_text(query)

    query_vector = tfidf_vectorizer.transform(
        [preprocessed_query]
    )

    similarity_scores = cosine_similarity(
        query_vector, tfidf_matrix
    ).flatten()

    search_results = []

    for i, score in enumerate(similarity_scores):
        search_results.append((
            list(documents.keys())[i],
            documents[list(documents.keys())[i]],
            score
        ))

    search_results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    return search_results


search_results = search(query, tfidf_matrix, tfidf_vectorizer)

print("\nQuery:", query)

for i, result in enumerate(search_results, start=1):
    print(f"\nRank: {i}")
    print("Document ID:", result[0])
    print("Document:", result[1])
    print("Similarity Score:", result[2])
    print("----------------------")

highest_rank_score = max(
    result[2] for result in search_results
)

print("The highest rank cosine score is:", highest_rank_score)
```
**Output:**<br>
<img width="767" height="718" alt="Screenshot 2026-11-03 090637" src="https://github.com/user-attachments/assets/d057b477-271a-405d-b180-d8dc1f0f9b57" />

**Result:**
Information Retrieval Using Vector Space Model in Python is implemented.
