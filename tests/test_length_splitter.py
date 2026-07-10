from vn_legal_rag.chunking import (
    LengthSplitter,
)

from vn_legal_rag.models import Chunk


text = "\n\n".join(

    f"Paragraph {i}\n"

    + ("ABC " * 80)

    for i in range(30)

)

chunk = Chunk(

    chunk_id="1",

    document_id=1,

    article="Điều 1",

    clause=None,

    point=None,

    chunk_index=0,

    sub_chunk_index=0,

    start_char=0,

    end_char=len(text),

    title="Test",

    legal_type="",

    legal_sectors="",

    issuing_authority="",

    issuance_date="",

    text=text,
)

splitter = LengthSplitter()

chunks = list(

    splitter.split(chunk)

)

print()

print(

    "Chunks:",

    len(chunks),

)

for c in chunks:

    print()

    print("=" * 80)

    print(c.chunk_id)

    print(len(c.text))

    print(c.text[:300])