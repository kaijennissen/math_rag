"""Main takeaway from v1 is, that calculating bm25 required a lot of
informations (number of documents, average length of the documents, etc.)
 to be calculated repeatedly.
All these informations could also be precalculated and stored alongside the
inverted index. While this could be done by further nesting dictionaries, I think
a more object-oriented approach (using lightweight dataclasses) is more convincing.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class Document:
    content: str
    terms: list[str] = field(init=False)
    id: int = field(init=False)
    length: int = field(init=False)
    term_count: Counter = field(init=False)

    def __post_init__(self):
        self.terms = self.content.split()
        self.id = hash(self.content)
        self.length = len(self.terms)  # derive from term_count
        self.term_count = Counter(self.terms)


@dataclass
class InvertedIndexEntry:
    """Class for holding a single entry in the Inverted Index"""

    doc_id: int
    count: int


def idf(n_q: int, N: int) -> float:
    nominator = N - n_q + 0.5
    denominator = n_q + 0.5
    return math.log(nominator / denominator + 1)


def term_frequency(
    f_q_D: int, doc_length: int, avg_doc_len: float, k: float, b: float
) -> float:
    nominator = f_q_D * (k + 1)
    denominator = f_q_D + k * (1 - b + b * doc_length / avg_doc_len)
    return nominator / denominator


@dataclass
class InvertedIndex:
    """Inverted Index class that stores additional meta-informations."""

    number_of_documents: int = 0
    average_document_length: float = 0
    entries: defaultdict[str, list[dict]] = field(
        default_factory=lambda: defaultdict(list)
    )
    documents: set = field(default_factory=set)

    def add_document(self, doc: Document):
        if doc.id not in self.documents:
            self.average_document_length = (
                self.average_document_length * self.number_of_documents + doc.length
            ) / (self.number_of_documents + 1)
            self.number_of_documents += 1
            for term, term_count in doc.term_count.items():
                self.entries[term].append({doc.id: term_count})  # missing
            self.documents.update({doc.id})
        else:
            print("doc alreday in the index")

    def add_documents(self, docs: list[Document]):
        for doc in docs:
            self.add_document(doc)

    def bm25_single_doc(
        self, query: str, doc: Document, k: float = 1.2, b: float = 0.8
    ) -> float:
        query_terms = query.split()

        term_score = []
        for term in query_terms:
            n_q = len(self.entries.get(term, []))  # number of docs containing the term
            term_idf = idf(n_q, self.number_of_documents)
            f_q_D = doc.term_count.get("term", 0)
            doc_length = doc.length
            term_freq = term_frequency(
                f_q_D=f_q_D,
                doc=doc_length,
                avg_doc_len=self.average_document_length,
                k=k,
                b=b,
            )
            term_score.append(term_idf * term_freq)
        score = sum(term_score)
        return score

    def bm25(self, query: str) -> list:
        pass


doc = Document("the cat sat on the mat")
inv_index = InvertedIndex()
inv_index.add_document(doc)
doc_list = [Document("the cat sat on the mat"), Document("the dog sat on the log")]
inv_index.add_documents(doc_list)
inv_index

if __name__ == "__main__":
    documents = [
        "the cat in the hat",
        "the cat sat on the mat",
        "the dog sat on the log",
    ]
    query = "what the cat"
