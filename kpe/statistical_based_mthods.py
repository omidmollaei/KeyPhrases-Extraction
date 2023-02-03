"""
Implement Some Statistical-Based approaches for extracting keywords and key phrases.
"""

import os
import re
import yake
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class TFIDF:
    """TF-IDF is a very  method for extracting keywords. For using this method, we need some
     training docs to extract some statistical data. TF-IDF is fast and easy to understand."""
    def __init__(self, path_to_docs, drop_threshold: float = 0.6, max_features: int = 20_000):
        """
        :param path_to_docs: path to text files. Each file indicates one document.
        :param drop_threshold: words that occures in documents more than this rate, will be removed at all.
        :param max_features: maximum number of count vectorizers features.
        """
        self.all_docs_name = os.listdir(path_to_docs)
        self.all_docs_name = [os.path.join(path_to_docs, dn) for dn in self.all_docs_name]
        self.all_docs = []

        # read document one by one, clea them and add to list.
        print("reading documents ...")
        for d in tqdm(self.all_docs_name):
            with open(d, 'r', encoding="utf8") as f:
                doc = f.read()
                doc = self._clean(doc)
                self.all_docs.append(doc)

        self.count_vec = CountVectorizer(max_df=drop_threshold, max_features=max_features)
        self.docs_count_vectorized = self.count_vec.fit_transform(self.all_docs)
        self.tf_idf = TfidfTransformer(smooth_idf=True, use_idf=True)
        self.tf_idf.fit(self.docs_count_vectorized)

    def _prepare(self, doc: str):
        if isinstance(doc, str):
            doc = [doc]
        vectorized = self.count_vec.transform(doc)
        tfidf = self.tf_idf.transform(vectorized)
        return tfidf.tocoo()

    def _keywords(self, v, count: int):
        if isinstance(v, list):
            vector = v[0]
        feature_names = self.count_vec.get_feature_names_out()
        keywords = []
        for col_n, value in zip(v.col, v.data):
            keywords.append([value, feature_names[col_n]])
        keywords = sorted(keywords, reverse=True)
        return np.array(keywords[:count])[:, ::-1]

    def _dec_prod_2(self, ref_set: list):
        result = set()
        for w1 in ref_set:
            for w2 in ref_set:
                result.add(w1 + " " + w2)
        return result

    def _dec_prod_3(self, ref_set: list):
        result = set()
        for w1 in ref_set:
            for w2 in ref_set:
                for w3 in ref_set:
                    result.add(w1 + " " + w2 + " " + w3)
        return result

    def _build_ngrams(self, doc, uni_grams: list):
        all_keywords = pd.Series(dict(uni_grams))
        all_keywords = dict(all_keywords.map(float))
        just_uni_grams = list(all_keywords.keys())
        all_bi_grams = self._dec_prod_2(just_uni_grams)
        uni_grams, bi_grams, tri_grams = all_keywords.copy(), dict(), dict()
        for b in all_bi_grams:
            if b in doc:
                b1, b2 = b.split()
                all_keywords[b] = all_keywords[b1] + all_keywords[b2]
                bi_grams[b] = all_keywords[b1] + all_keywords[b2]
        all_tri_grams = self._dec_prod_3(just_uni_grams)
        for t in all_tri_grams:
            if t in doc:
                t1, t2, t3 = t.split()
                all_keywords[t] = all_keywords[t1] + all_keywords[t2] + all_keywords[t3]
                tri_grams[t] = all_keywords[t1] + all_keywords[t2] + all_keywords[t3]
        return uni_grams, bi_grams, tri_grams

    def extract(self, doc: str, top: int = 5):
        doc_cleaned = self._clean(doc)
        vectors = self._prepare(doc_cleaned)
        uni_grams = self._keywords(vectors, count=top*5)
        uni_, bi_, tri_ = self._build_ngrams(self._clean(doc, remove_stopwords=False), uni_grams)
        tri_ = pd.Series(tri_).sort_values(ascending=False).iloc[:2*2]
        bi_ = pd.Series(bi_).sort_values(ascending=False).iloc[:3*2]
        uni_ = pd.Series(uni_).sort_values(ascending=False).iloc[:(top - (len(tri_) + len(bi_)))*2]
        keywords = pd.Series({**uni_, **bi_, **tri_}).sort_values(ascending=False).iloc[:top]
        return keywords

    def _clean(self, doc, remove_stopwords: bool = True):
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


class Yake:
    def __init__(
            self,
            lang: str = "en",
            dedup_lim: float = 0.75,
            top: int = 5,
            max_n_gram_size: int = 3):

        self.extractor = yake.KeywordExtractor(lan=lang, dedupLim=dedup_lim, top=top, n=max_n_gram_size)

    def extract(self, document: str):
        keywords = self.extractor.extract_keywords(document)
        return dict(keywords)

    @staticmethod
    def preprocess(document: str):
        def _clean(sent):
            sent = sent.lower().strip()  # lowercase the document
            sent = re.sub('[^a-z\d ]', '', sent)  # remove non-english characters
            sent = re.sub(' +', ' ', sent)  # Remove extra white spaces
            return sent

        sentences = nltk.sent_tokenize(document)
        sentences_cleaned = [_clean(s) for s in sentences]
        document = " . ".join(sentences_cleaned)
        return document

    @classmethod
    def extract_keywords(cls, document: str, lang: str = "en", dedup_lim: float = 0.75, top: int = 5,
                         ngrams: int = 3):

        document_preprocessed = cls.preprocess(document)
        extractor = yake.KeywordExtractor(lan=lang, dedupLim=dedup_lim, top=top, n=ngrams)
        keywords = extractor.extract_keywords(document)
        return dict(keywords)
