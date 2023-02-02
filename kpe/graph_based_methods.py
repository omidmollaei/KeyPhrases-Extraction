"""
Implement Some Graph-Based approaches for extracting keywords and key phrases.
"""

import re
import nltk
import numpy as np
import pandas as pd
import yake

from FRAKE import FRAKE
from typing import List, Tuple
from collections import namedtuple


class Node:
    """
    An instance of this class represents a node of a graph. Each node
    itself has a score and it's connections to other nodes have weights.
    So a node has one score and multiple weights (with each connections).
    """

    def __init__(self, name: str, score: int = 1):
        self.name = name
        self.score = score
        self.connections = list()
        self.connections_with_weights = None
        self.connections_name = None
        self.connections_weight = None

    def build_connections(self, nodes: List[Tuple]):
        """This method builds the connections of the node.
        Arg:
            nodes:It is a list of tuples, where each tuple indicates one connection
                  The first items of the tuple is the connected node itself, and the
                  second item of the tuple is the connection weight.
        """
        self.connections = [node for node, weight in nodes]
        self.connections_with_weights = nodes
        self.connections_name = [node.name for node, w in nodes]
        self.connections_weight = [w for node, w in nodes]

    def add_connection(self, node, weight: int = 1):
        """Add a new connection to node"""
        self.connections.append((node, weight))

    def remove_connection(self, node_name: str):
        """Remove the connection to another node with the name of that specific node."""
        for i, conn in self.connections:
            if node_name == conn.name:
                del self.connections[i]
                break

    def node_info(self):
        info = f"[\n  Name: {self.name}\n  Score: "
        info = info + f"{self.score}\n  Connections: {self.connections_name}\n]"
        return info


class Graph:
    def __init__(self, nodes: List[Node], name: str = "Graph"):
        """The abstract class, representing a graph. """
        self.all_nodes = {node.name: node for node in nodes}
        self.NodeInfo = namedtuple("NodeInfo", "name connections score")

    def __call__(self, node_name):
        return self.NodeInfo(
            node_name,
            self.all_nodes[node_name].connections,
            self.all_nodes[node_name].score,
        )


class TextRank:
    def __init__(self,
                 window_size: int = 2,
                 pos_filters: Tuple = ("JJ", "NN", "NNP", "NNS", "VBG", "VBN", "VB")):
        """Setup some config for later extraction of keywords/key phrases.
        Args:
            window_size: window size of co-occur.
            pos_filters: syntactic filter for uni-grams selection (as candidates).
        """
        self.filters = pos_filters
        self.window_size = window_size

    def _uni_gram_candidates(self, pos_tags):
        uni_gram_candidates = list()
        for word, pos in pos_tags:
            if pos in self.filters:
                uni_gram_candidates.append(word)
        return uni_gram_candidates

    def _find_connections(self, document_tokenized, uni_gram_candidates):
        connections = {candidate: set() for candidate in uni_gram_candidates}
        # Slice over document to detect nodes and their edges
        for w_begin in range(0, len(document_tokenized) - (self.window_size - 1)):
            w_end = w_begin + self.window_size
            window = document_tokenized[w_begin:w_end]
            for i, w1 in enumerate(window[:-1]):
                for j, w2 in enumerate(window[i + 1:]):
                    if w1 in uni_gram_candidates and w2 in uni_gram_candidates:
                        connections[w1].add(w2)
                        connections[w2].add(w1)
        return connections

    def _build_nodes(self, document_tokenized, uni_gram_candidates):
        """Find the connections (edges) and then build their corresponding object"""
        raw_connections = self._find_connections(
            document_tokenized, uni_gram_candidates)  # connections as set
        raw_nodes = {c: Node(c, 1) for c in uni_gram_candidates}  # Build nodes without connections.
        nodes = []
        for node_name, connections in raw_connections.items():
            node_connections = [(raw_nodes[c], 1) for c in connections]
            node = raw_nodes[node_name]
            node.build_connections(node_connections)
            nodes.append(node)
        return nodes

    def _ranking(self, graph: Graph, d, n_iter):
        """Here, we run modified page rank algorithm"""
        scores = {node_name: node_obj.score for node_name, node_obj in graph.all_nodes.items()}
        for iter_no in range(n_iter):
            for node_name, node_obj in graph.all_nodes.items():
                new_score = 0  # Here we do not need connections weight at all.
                for connected_node in node_obj.connections:
                    new_score += (connected_node.score / len(connected_node.connections))
                graph.all_nodes[node_name].socre = (1 - d) + d * new_score
                scores[node_name] = (1 - d) + d * new_score
        return scores

    def _build_n_grams(self, document_tokenized: list, uni_grams: dict, ngrams: int):
        extended_keywords = uni_grams.copy()
        for w_begin in range(0, len(document_tokenized) - (ngrams - 1)):  # slice over document
            w_end = w_begin + ngrams
            window = document_tokenized[w_begin: w_end]
            for i, word1 in enumerate(window[:-1]):
                for j, word2 in enumerate(window[i + 1:]):
                    if (word1 in uni_grams) and (word2 in uni_grams):
                        extended_keywords[word1 + " " + word2] = uni_grams[word1] + uni_grams[word2]
        extended_keywords = pd.Series(extended_keywords).sort_values(ascending=False)
        return extended_keywords

    # def _post_processing(self, keywords: pd.Series, max_len: int):

    def extract(self,
                document: str,
                d: float = 0.85,
                n_iter: int = 20,
                top: int = 6,
                max_len: int = 2,):

        """This assumes that the input document has been already preprocessed."""
        document_tokenized = nltk.word_tokenize(document)
        pos_tags = nltk.pos_tag(document_tokenized)
        uni_gram_candidates = self._uni_gram_candidates(pos_tags)
        nodes = self._build_nodes(document_tokenized, uni_gram_candidates)
        graph = Graph(nodes, name="document_graph")
        scores = self._ranking(graph, d, n_iter)
        scores = pd.Series(scores).sort_values(ascending=False)
        selected_uni_grams = scores.iloc[:min(int(top + (top // 2)), len(scores))]
        selected_uni_grams = dict(selected_uni_grams)
        keywords = self._build_n_grams(document_tokenized, selected_uni_grams, max_len)
        # if post_processing:
        #    keywords = self._post_processing(keywords, max_len)
        keywords = keywords.iloc[:top]
        return keywords

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
    def extract_keywords(cls, document: str):
        document_preprocessed = cls.preprocess(document)
        extractor = cls()
        keywords = extractor.extract(document_preprocessed)
        return keywords


class SingleRank(TextRank):
    def _dec_prod(self, window):
        result = set()
        for i, word1 in enumerate(window[:-1]):
            for word2 in window[i + 1:]:
                result.add((word1, word2))
        return result

    def _edges_weight(self, graph: Graph, document_tokenized: str, uni_grams: list):
        for w_begin in range(0, len(document_tokenized) - (self.window_size * 2 - 1)):
            w_end = w_begin + self.window_size * 2
            window = document_tokenized[w_begin:w_end]
            pairs_in_window = self._dec_prod(window=window)
            for pair in pairs_in_window:
                w1, w2 = pair
                if not ((w1 in uni_grams) and (w2 in uni_grams)):
                    continue
                node1, node2 = graph.all_nodes[w1], graph.all_nodes[w2]
                if w2 in node1.connections_name:
                    distance = abs(window.index(w2) - window.index(w1))
                    prev_weight_i = node1.connections_name.index(w2)
                    prev_weight_j = node2.connections_name.index(w1)
                    prev_weight = node1.connections_weight[prev_weight_i]
                    new_weight = prev_weight + (1 / distance)
                    node1.connections_weight[prev_weight_i] = new_weight
                    node2.connections_weight[prev_weight_j] = new_weight

    def _ranking(self, graph: Graph, d, n_iter):
        """Here, we run modified page rank algorithm"""
        scores = {node_name: node_obj.score for node_name, node_obj in graph.all_nodes.items()}
        for iter_no in range(n_iter):
            for node_name, node_obj in graph.all_nodes.items():
                new_score = 0  # Here we do not need connections weight at all.
                for connected_node in node_obj.connections:
                    denominator = sum(connected_node.connections_weight)
                    w_ji = graph.all_nodes[connected_node.name].connections_name.index(node_name)
                    w_ji = graph.all_nodes[connected_node.name].connections_weight[w_ji]
                    new_score += (connected_node.score * w_ji) / denominator
                graph.all_nodes[node_name].socre = (1 - d) + d * new_score
                scores[node_name] = (1 - d) + d * new_score
        return scores

    def extract(self,
                document: str,
                d: float = 0.85,
                n_iter: int = 20,
                top: int = 6,
                max_len: int = 2,
                post_processing: bool = True):

        """This assumes that the input document has been already preprocessed."""
        document_tokenized = nltk.word_tokenize(document)
        pos_tags = nltk.pos_tag(document_tokenized)
        uni_gram_candidates = self._uni_gram_candidates(pos_tags)
        nodes = self._build_nodes(document_tokenized, uni_gram_candidates)
        graph = Graph(nodes, name="document_graph")
        self._edges_weight(graph, document_tokenized, uni_gram_candidates)
        scores = self._ranking(graph, d, n_iter)
        scores = pd.Series(scores).sort_values(ascending=False)
        # selected_uni_grams = scores.iloc[:min(int(top+(top//2)), len(scores))]
        selected_uni_grams = dict(scores)
        keywords = self._build_n_grams(document_tokenized, selected_uni_grams, max_len)
        # if post_processing:
        #    keywords = self._post_processing(keywords, max_len)
        keywords = keywords.iloc[:top]
        return keywords


class Frake:
    """
    Farke is another graph-based method for extracting keywords and key phrases of a document.
    It can be used to extract keywords of persian language directly. The language can be specified
    in constructor.
    Frake source:
        https://github.com/cominsys/FRAKE
    """
    def __init__(self, lang: str = "en"):
        self.lang = lang

    def extract(self, document: str, hu_hiper: float = 0.4, top: int = 5):
        """It is assumed that the input document has been preprocessed."""
        extractor = FRAKE.KeywordExtractor(lang='en', hu_hiper=hu_hiper, Number_of_keywords=top)
        keywords = extractor.extract_keywords(document)
        return keywords

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
    def extract_keywords(cls, document: str):
        document_preprocessed = cls.preprocess(document)
        extractor = FRAKE.KeywordExtractor(lang='en', hu_hiper=0.4, Number_of_keywords=5)
        keywords = extractor.extract_keywords(document)
        return keywords


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
