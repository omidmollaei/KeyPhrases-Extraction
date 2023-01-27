"""
Implement Some Graph-Based approaches for extracting keywords and key phrases.
"""

import re
import nltk
import numpy as np
import pandas as pd
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
