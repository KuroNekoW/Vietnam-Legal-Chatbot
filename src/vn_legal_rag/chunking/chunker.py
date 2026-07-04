from vn_legal_rag.chunking.legalsplitter import LegalSplitter
from vn_legal_rag.chunking.lengthsplitter import LengthSplitter

class LegalChunker:

    def __init__(self):

        self.semantic = LegalSplitter()

        self.length = LengthSplitter()

    def chunk(self, document):

        for chunk in self.semantic.split(document):

            yield from self.length.split(chunk)