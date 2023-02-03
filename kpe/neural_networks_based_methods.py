"""
Using pretrained language-models to extract keywords and key-phrases of documents.
"""

import re
import itertools
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


class KPESentenceTransformer:
    def __init__(self, ngrams: tuple = (1, 2), model_name: str = "distilbert-base-nli-mean-tokens"):
        self.count_vectorizer = CountVectorizer(ngram_range=ngrams, stop_words="english")
        print(f"loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def _clean(self, doc: str, remove_stopwords: bool = False):
        """1. Lower case the text
        2. Remove everything's except english alphabet (keywords are words)
        3. Remove extra white spaces.
        4. Remove stop-words"""

        doc = doc.lower()
        doc = re.sub('[^a-z ]', ' ', doc)
        doc = re.sub('  +', ' ', doc).strip()
        if not remove_stopwords:
            return doc
        words = [w for w in word_tokenize(doc) if w not in set(stopwords.words('english'))]
        return ' '.join(words)

    def max_sum_sim(self, candidates, doc_embedding, candidate_embeddings, top_n, nr_candidates):
        # Calculate distances and extract keywords
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        distances_candidates = cosine_similarity(candidate_embeddings, candidate_embeddings)

        # Get top_n words as candidates based on cosine similarity
        words_idx = list(distances.argsort()[0][-nr_candidates:])
        words_vals = [candidates[index] for index in words_idx]
        distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

        # Calculate the combination of words that are the least similar to each other
        min_sim = np.inf
        candidate = None
        for combination in itertools.combinations(range(len(words_idx)), top_n):
            sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
            if sim < min_sim:
                candidate = combination
                min_sim = sim

        return [words_vals[idx] for idx in candidate]

    def extract(self, doc: str, top: int = 5):
        doc_cleaned = self._clean(doc)
        self.count_vectorizer.fit([doc_cleaned])
        candidates = self.count_vectorizer.get_feature_names_out()
        doc_embedding = self.model.encode([doc])
        candidates_embedding = self.model.encode(candidates)
        keywords = self.max_sum_sim(candidates, doc_embedding, candidates_embedding, top, top*3)
        return keywords



