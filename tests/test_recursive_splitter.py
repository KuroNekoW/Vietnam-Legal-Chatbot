from vn_legal_rag.chunking import LengthSplitter
from vn_legal_rag.models import Chunk


def create_dummy_chunk():

    text = "abc " * 1500

    return Chunk(
        chunk_id="1",
        document_id=1,
        article="Điều 99",
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        title="Dummy",
        legal_type="Luật",
        legal_sectors="Test",
        issuing_authority="ABC",
        issuance_date="2026",
        text=text,
    )


chunk = create_dummy_chunk()

splitter = LengthSplitter(
    chunk_size=1000,
    overlap=200,
)

chunks = list(
    splitter.split(chunk)
)

print()

print("Total:", len(chunks))

print()

for c in chunks:

    print("=" * 80)

    print(c.chunk_id)

    print(c.start_char)

    print(c.end_char)

    print(len(c.text))