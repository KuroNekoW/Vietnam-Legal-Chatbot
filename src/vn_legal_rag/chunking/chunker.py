from vn_legal_rag.chunking.splitter import LegalSplitter


class LegalChunker:

    def __init__(self):

        self.splitter = LegalSplitter()

    def chunk(self, document):

        yield from self.splitter.split(document)