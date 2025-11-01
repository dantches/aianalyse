import ast
from typing import List, Dict

def extract_ast_graph_python(code: str, max_nodes: int = 512):
    """
    Возвращает:
     - tokens: список токенов/имён
     - adj: adjacency matrix in sparse form (list of (i,j) edges)
    Простейшая версия: AST nodes; коннект родитель->дети и siblings.
    """
    tree = ast.parse(code)
    nodes = []
    edges = []

    def visit(node, parent_idx=None):
        idx = len(nodes)
        nodes.append(type(node).__name__)
        if parent_idx is not None:
            edges.append((parent_idx, idx))
            edges.append((idx, parent_idx))
        for child in ast.iter_child_nodes(node):
            visit(child, idx)

    visit(tree, None)
    # tokens fallback: node type names
    tokens = nodes[:max_nodes]
    adj = edges[:max_nodes*4]
    return tokens, adj
