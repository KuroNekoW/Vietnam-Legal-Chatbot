from vn_legal_rag.models import LegalDocument

doc = LegalDocument(
    id=1,
    title="Luật Đất đai",
    content="Hello"
)

print(doc.model_dump())
print(type(doc))