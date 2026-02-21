"""
The goal of this script is to build a full-text search engine in pure python.
As the goal is to get a better conceptual understanding, every line is written by
myself.
"""

import math
from collections import Counter, defaultdict

documents = ["the cat in the hat", "the cat sat on the mat", "the dog sat on the log"]

inv_index = defaultdict(list)
for doc_num, doc in enumerate(documents, start=1):
    doc_tokens = Counter(doc.split())
    for token, token_count in doc_tokens.items():
        inv_index[token].append({"doc": doc_num, "count": token_count})


def idf(n_q: int, N: int) -> float:
    nominator = N - n_q + 0.5
    denominator = n_q + 0.5
    return math.log(nominator / denominator + 1)


def term_frequency(
    q: str, doc: list[str], avg_doc_len: float, k: float, b: float
) -> float:
    f_q_D = doc.count(q)
    nominator = f_q_D * (k + 1)
    denominator = f_q_D + k * (1 - b + b * len(doc) / avg_doc_len)
    return nominator / denominator


def get_term_count(term: str, inv_index: dict) -> int:
    return len(inv_index.get(term, []))


def bm25(
    inv_index: dict,
    query: str,
    doc: list,
    N: int,
    avg_doc_len: float,
    k: float = 1.2,
    b: float = 0.75,
) -> float:
    query_terms = query.split()

    term_score = []
    for term in query_terms:
        n_q = get_term_count(term, inv_index)
        term_idf = idf(n_q, N)
        term_freq = term_frequency(q=term, doc=doc, avg_doc_len=avg_doc_len, k=k, b=b)
        term_score.append(term_idf * term_freq)

    score = sum(term_score)
    return score


if __name__ == "__main__":
    query = "what the cat"
