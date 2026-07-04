from vn_legal_rag.chunking.splitter import LegalSplitter
from vn_legal_rag.chunking.recursive_splitter import LengthSplitter

class LegalChunker:

    def __init__(self):

        self.semantic = LegalSplitter()

        self.length = LengthSplitter()

    def chunk(self, document):

        for chunk in self.semantic.split(document):

            yield from self.length.split(chunk)