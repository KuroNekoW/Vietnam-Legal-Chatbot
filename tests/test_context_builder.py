from __future__ import annotations

from vn_legal_rag.context import ContextBuilder
from vn_legal_rag.retrieval.retriever import RetrievedChunk


def make_chunk(
    *,
    chunk_id: str,
    document_id: int,
    article_no: int,
    article: str,
    clause_no: int | None,
    clause: str | None,
    point_no: str | None,
    point: str | None,
    text: str,
    title: str = "Bộ luật Lao động 2019",
    legal_type: str = "Bộ luật",
    legal_sectors: str = "Lao động",
    issuing_authority: str = "Quốc hội",
    issuance_date: str = "2019-11-20",
    url: str = "https://example.com",
    signers: str = "Nguyễn Thị Kim Ngân",
    chunk_index: int = 0,
    sub_chunk_index: int = 0,
    rerank_score: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=0.90,
        source_query="test query",

        document_id=document_id,

        article=article,
        article_no=article_no,

        clause=clause,
        clause_no=clause_no,

        point=point,
        point_no=point_no,

        chunk_index=chunk_index,
        sub_chunk_index=sub_chunk_index,

        start_char=0,
        end_char=len(text),

        title=title,
        legal_type=legal_type,
        legal_sectors=legal_sectors,
        issuing_authority=issuing_authority,
        issuance_date=issuance_date,

        url=url,
        signers=signers,

        text=text,

        rerank_score=rerank_score,
    )


def test_empty_input():
    builder = ContextBuilder()

    result = builder.build([])

    assert result.is_empty
    assert result.input_chunk_count == 0
    assert result.deduplicated_chunk_count == 0
    assert result.document_count == 0
    assert result.rendered_text == ""


def test_groups_chunks_by_document():
    builder = ContextBuilder()

    chunks = [
        make_chunk(
            chunk_id="doc1-a36-c1",
            document_id=1,
            article_no=36,
            article="Điều 36. Quyền đơn phương chấm dứt hợp đồng lao động",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Người lao động có quyền đơn phương chấm dứt hợp đồng lao động.",
            chunk_index=0,
            rerank_score=-0.2,
        ),
        make_chunk(
            chunk_id="doc2-a10-c1",
            document_id=2,
            article_no=10,
            article="Điều 10. Quy định chung",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Nội dung của văn bản thứ hai.",
            chunk_index=0,
            rerank_score=-0.3,
            title="Nghị định 01/2025/NĐ-CP",
            legal_type="Nghị định",
        ),
    ]

    result = builder.build(chunks)

    assert result.document_count == 2
    assert len(result.documents) == 2

    assert result.documents[0].document_id == 1
    assert result.documents[1].document_id == 2


def test_groups_chunks_by_article():
    builder = ContextBuilder()

    chunks = [
        make_chunk(
            chunk_id="a36-c2",
            document_id=1,
            article_no=36,
            article="Điều 36. Quyền đơn phương chấm dứt hợp đồng lao động",
            clause_no=2,
            clause="Khoản 2",
            point_no=None,
            point=None,
            text="Nội dung khoản 2.",
            chunk_index=0,
            rerank_score=-0.1,
        ),
        make_chunk(
            chunk_id="a36-c1",
            document_id=1,
            article_no=36,
            article="Điều 36. Quyền đơn phương chấm dứt hợp đồng lao động",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Nội dung khoản 1.",
            chunk_index=0,
            rerank_score=-0.2,
        ),
        make_chunk(
            chunk_id="a37-c1",
            document_id=1,
            article_no=37,
            article="Điều 37. Quyền đơn phương chấm dứt hợp đồng lao động của người sử dụng lao động",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Nội dung điều 37.",
            chunk_index=0,
            rerank_score=-0.3,
        ),
    ]

    result = builder.build(chunks)

    assert len(result.documents) == 1

    document = result.documents[0]

    assert len(document.articles) == 2

    assert document.articles[0].article_no == 36
    assert document.articles[1].article_no == 37


def test_sorts_clause_in_legal_order():
    builder = ContextBuilder()

    chunks = [
        make_chunk(
            chunk_id="c3",
            document_id=1,
            article_no=36,
            article="Điều 36. Quy định",
            clause_no=3,
            clause="Khoản 3",
            point_no=None,
            point=None,
            text="Nội dung khoản 3.",
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="c1",
            document_id=1,
            article_no=36,
            article="Điều 36. Quy định",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Nội dung khoản 1.",
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="c2",
            document_id=1,
            article_no=36,
            article="Điều 36. Quy định",
            clause_no=2,
            clause="Khoản 2",
            point_no=None,
            point=None,
            text="Nội dung khoản 2.",
            chunk_index=0,
        ),
    ]

    result = builder.build(chunks)

    article = result.documents[0].articles[0]

    assert [
        chunk.clause_no
        for chunk in article.chunks
    ] == [1, 2, 3]


def test_sorts_articles_numerically():
    builder = ContextBuilder()

    chunks = [
        make_chunk(
            chunk_id="a10",
            document_id=1,
            article_no=10,
            article="Điều 10. Mười",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Điều 10.",
        ),
        make_chunk(
            chunk_id="a2",
            document_id=1,
            article_no=2,
            article="Điều 2. Hai",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Điều 2.",
        ),
        make_chunk(
            chunk_id="a1",
            document_id=1,
            article_no=1,
            article="Điều 1. Một",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Điều 1.",
        ),
    ]

    result = builder.build(chunks)

    article_numbers = [
        article.article_no
        for article in result.documents[0].articles
    ]

    assert article_numbers == [1, 2, 10]


def test_exact_duplicates_are_removed():
    builder = ContextBuilder()

    chunks = [
        make_chunk(
            chunk_id="duplicate-1",
            document_id=1,
            article_no=36,
            article="Điều 36. Quy định",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Nội dung giống nhau.",
            rerank_score=-0.50,
        ),
        make_chunk(
            chunk_id="duplicate-2",
            document_id=1,
            article_no=36,
            article="Điều 36. Quy định",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="  Nội dung   giống nhau.  ",
            rerank_score=-0.20,
        ),
    ]

    result = builder.build(chunks)

    assert result.input_chunk_count == 2
    assert result.deduplicated_chunk_count == 1

    chunk = next(result.iter_chunks())

    assert chunk.chunk_id == "duplicate-2"
    assert chunk.rerank_score == -0.20


def test_equal_score_does_not_mean_duplicate():
    builder = ContextBuilder()

    chunks = [
        make_chunk(
            chunk_id="different-1",
            document_id=1,
            article_no=36,
            article="Điều 36. Quy định",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Quy định A.",
            rerank_score=-0.5,
        ),
        make_chunk(
            chunk_id="different-2",
            document_id=1,
            article_no=36,
            article="Điều 36. Quy định",
            clause_no=2,
            clause="Khoản 2",
            point_no=None,
            point=None,
            text="Quy định B.",
            rerank_score=-0.5,
        ),
    ]

    result = builder.build(chunks)

    assert result.deduplicated_chunk_count == 2
    assert len(list(result.iter_chunks())) == 2


def test_contextual_text_contains_hierarchy():
    builder = ContextBuilder()

    chunks = [
        make_chunk(
            chunk_id="ctx-1",
            document_id=1,
            article_no=36,
            article="Điều 36. Quyền đơn phương chấm dứt hợp đồng lao động",
            clause_no=3,
            clause="Khoản 3",
            point_no="a",
            point="a) Trường hợp thứ nhất",
            text="Người lao động phải báo trước.",
        ),
    ]

    result = builder.build(chunks)

    context_chunk = next(result.iter_chunks())

    assert (
        "Điều 36. Quyền đơn phương chấm dứt hợp đồng lao động"
        in context_chunk.contextual_text
    )

    assert "Khoản 3" in context_chunk.contextual_text
    assert "a) Trường hợp thứ nhất" in context_chunk.contextual_text
    assert "Người lao động phải báo trước." in context_chunk.contextual_text


def test_rendered_context_contains_document_metadata():
    builder = ContextBuilder()

    chunks = [
        make_chunk(
            chunk_id="meta-1",
            document_id=123,
            article_no=1,
            article="Điều 1. Phạm vi điều chỉnh",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Nội dung.",
            title="Bộ luật Lao động 2019",
            legal_type="Bộ luật",
            issuing_authority="Quốc hội",
            issuance_date="2019-11-20",
            signers="Nguyễn Thị Kim Ngân",
        ),
    ]

    result = builder.build(chunks)

    rendered = result.rendered_text

    assert "[VĂN BẢN]" in rendered
    assert "Bộ luật Lao động 2019" in rendered
    assert "Loại văn bản: Bộ luật" in rendered
    assert "Cơ quan ban hành: Quốc hội" in rendered
    assert "Ngày ban hành: 2019-11-20" in rendered
    assert "Người ký: Nguyễn Thị Kim Ngân" in rendered
    assert "Điều 1. Phạm vi điều chỉnh" in rendered
    assert "Khoản 1" in rendered
    assert "Nội dung." in rendered


def test_documents_can_be_limited():
    builder = ContextBuilder(
        max_documents=1,
    )

    chunks = [
        make_chunk(
            chunk_id="doc1",
            document_id=1,
            article_no=1,
            article="Điều 1",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Document 1",
            rerank_score=-0.1,
        ),
        make_chunk(
            chunk_id="doc2",
            document_id=2,
            article_no=1,
            article="Điều 1",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Document 2",
            rerank_score=-0.2,
        ),
    ]

    result = builder.build(chunks)

    assert result.document_count == 1


def test_articles_can_be_limited_per_document():
    builder = ContextBuilder(
        max_articles_per_document=1,
    )

    chunks = [
        make_chunk(
            chunk_id="a1",
            document_id=1,
            article_no=1,
            article="Điều 1",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Article 1",
        ),
        make_chunk(
            chunk_id="a2",
            document_id=1,
            article_no=2,
            article="Điều 2",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Article 2",
        ),
    ]

    result = builder.build(chunks)

    assert len(
        result.documents[0].articles
    ) == 1


def test_chunks_can_be_limited_per_article():
    builder = ContextBuilder(
        max_chunks_per_article=1,
    )

    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            document_id=1,
            article_no=1,
            article="Điều 1",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Chunk 1",
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-2",
            document_id=1,
            article_no=1,
            article="Điều 1",
            clause_no=2,
            clause="Khoản 2",
            point_no=None,
            point=None,
            text="Chunk 2",
            chunk_index=1,
        ),
    ]

    result = builder.build(chunks)

    assert len(
        result.documents[0].articles[0].chunks
    ) == 1


def test_different_documents_with_same_article_are_not_merged():
    builder = ContextBuilder()

    chunks = [
        make_chunk(
            chunk_id="doc1",
            document_id=1,
            article_no=1,
            article="Điều 1. Quy định",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Văn bản A.",
        ),
        make_chunk(
            chunk_id="doc2",
            document_id=2,
            article_no=1,
            article="Điều 1. Quy định",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Văn bản B.",
        ),
    ]

    result = builder.build(chunks)

    assert result.document_count == 2
    assert (
        result.documents[0].document_id
        != result.documents[1].document_id
    )


def test_rendered_article_does_not_repeat_article_heading_for_each_chunk():
    builder = ContextBuilder()

    chunks = [
        make_chunk(
            chunk_id="c1",
            document_id=1,
            article_no=36,
            article="Điều 36. Quy định",
            clause_no=1,
            clause="Khoản 1",
            point_no=None,
            point=None,
            text="Nội dung 1.",
        ),
        make_chunk(
            chunk_id="c2",
            document_id=1,
            article_no=36,
            article="Điều 36. Quy định",
            clause_no=2,
            clause="Khoản 2",
            point_no=None,
            point=None,
            text="Nội dung 2.",
        ),
    ]

    result = builder.build(chunks)

    rendered = result.documents[0].articles[0].rendered_text

    assert rendered.count(
        "Điều 36. Quy định"
    ) == 1

    assert "Nội dung 1." in rendered
    assert "Nội dung 2." in rendered