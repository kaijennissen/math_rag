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
    doc_ids: set = field(default_factory=set)
    documents: list = field(default_factory=list)

    def add_document(self, doc: Document):
        if doc.id not in self.doc_ids:
            self.average_document_length = (
                self.average_document_length * self.number_of_documents + doc.length
            ) / (self.number_of_documents + 1)
            self.number_of_documents += 1
            for term, term_count in doc.term_count.items():
                self.entries[term].append({doc.id: term_count})  # missing
            self.doc_ids.update({doc.id})
            self.documents.extend([doc])
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
            f_q_D = doc.term_count.get(term, 0)
            doc_length = doc.length
            term_freq = term_frequency(
                f_q_D=f_q_D,
                doc_length=doc_length,
                avg_doc_len=self.average_document_length,
                k=k,
                b=b,
            )
            term_score.append(term_idf * term_freq)
        score = sum(term_score)
        return score

    def bm25(self, query: str, k: float = 1.2, b: float = 0.8) -> list:
        return sorted(
            [
                self.bm25_single_doc(query=query, doc=doc, k=k, b=b)
                for doc in self.documents
            ],
            reverse=True,
        )


if __name__ == "__main__":
    documents = [
        "Im folgenden sei $\\underline{X}=(X, \\mathcal{T})$ ein topologischer Raum und $\\mathcal{A}, \\mathcal{B}, \\ldots$ seien Mengen von Teilmengen von $X$, also $\\mathcal{A}, \\mathcal{B}, \\ldots \\subset \\mathcal{P} X$.\n\nBereits vor der Einführung des Begriffes der $F_{\\sigma}$-Mengen haben wir (in 1.2.4) bemerkt, dass die Vereinigung unendlich vieler abgeschlossener Mengen in $\\underline{X}$ nicht notwendig abgeschlossen in $\\underline{X}$ ist. Unter gewissen Bedingungen kann man jedoch ein positives Resultat herleiten, wie wir jetzt zunächst zeigen wollen.",  # noqa E501
        "(1) $\\underline{X}$ heißt $T_{3}$-Raum, wenn zu jedem Punkt $x$ von $\\underline{X}$ und jeder abgeschlossenen Menge $A$ in $\\underline{X}$ mit $x \\notin A$ (offene) Umgebungen $U$ von $x$ und $V$ von $A$ in $\\underline{X}$ mit $U \\cap V=\\emptyset$ existieren.\n\n(2) $\\underline{X}$ heißt regulär, wenn $\\underline{X}$ gleichzeitig $\\mathrm{T}_{3}$-Raum und $\\mathrm{T}_{1}$-Raum ist.",  # noqa: E501
        "(1) $\\underline{X}$ heißt $T_{4}$-Raum, wenn zu je zwei disjunkten abgeschlossenen Mengen $A$ und $B$ in $\\underline{X}$ (offene) Umgebungen $U$ von $A$ und $V$ von $B$ mit $U \\cap V=\\emptyset$ existieren.\n\n(2) $\\underline{X}$ heißt normal, wenn $\\underline{X}$ gleichzeitig $\\mathrm{T}_{4}$-Raum und $\\mathrm{T}_{1}$-Raum ist.",  # noqa: E501
    ]
    query = "Was ist ein T4-Raum?"
    # add pylatexenc + replace unicode strings (subset etc.)
    inv_index = InvertedIndex()
    doc_list = [Document(x) for x in documents]
    inv_index.add_documents(doc_list)
    bm25_score = inv_index.bm25(query=query)
    print(bm25_score)
