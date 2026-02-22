"""Main takeaway from v1 is, that calculating bm25 required a lot of
informations (number of documents, average length of the documents, etc.)
 to be calculated repeatedly.
All these informations could also be precalculated and stored alongside the
inverted index. While this could be done by further nesting dictionaries, I think
a more object-oriented approach (using lightweight dataclasses) is more convincing.
"""

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

    def bm25(self, query: str) -> list[dict[float, str]]:
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
