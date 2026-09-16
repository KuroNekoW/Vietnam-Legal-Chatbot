from __future__ import annotations

from vn_legal_rag.generation import (
    LegalAnswer,
    build_system_prompt,
    build_user_prompt,
)


def test_answer_schema_valid():
    answer = LegalAnswer(
        answer=(
            "Người lao động làm việc theo hợp đồng lao động "
            "không xác định thời hạn phải báo trước ít nhất 45 ngày."
        ),
        citations=[],
        insufficient_evidence=False,
        note=None,
    )

    assert answer.answer
    assert answer.insufficient_evidence is False

    print("\n[PASS] test_answer_schema_valid")
    print(answer.model_dump_json(indent=2))


def test_answer_schema_rejects_extra_fields():
    try:
        LegalAnswer(
            answer="Test",
            citations=[],
            insufficient_evidence=False,
            note=None,
            hallucinated_field="invalid",
        )
    except Exception:
        print(
            "\n[PASS] "
            "test_answer_schema_rejects_extra_fields"
        )
        return

    raise AssertionError(
        "Schema accepted an unexpected field."
    )


def test_answer_schema_supports_citation():
    answer = LegalAnswer(
        answer="Phải báo trước ít nhất 45 ngày.",
        citations=[
            {
                "chunk_id": "chunk-12345",
                "document_id": 123,
                "title": "Bộ luật Lao động",
                "article": "Điều 37",
                "clause": "Khoản 3",
                "point": None,
                "issuance_date": "31/12/2015",
            }
        ],
        insufficient_evidence=False,
        note=None,
    )

    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "chunk-12345"
    assert answer.citations[0].article == "Điều 37"
    assert answer.citations[0].clause == "Khoản 3"

    print(
        "\n[PASS] "
        "test_answer_schema_supports_citation"
    )


def test_system_prompt_exists():
    prompt = build_system_prompt()

    assert prompt
    assert "CHỈ trên" in prompt
    assert "Không" in prompt
    assert "insufficient_evidence" in prompt

    print(
        "\n[PASS] test_system_prompt_exists"
    )


def test_user_prompt_contains_query_and_context():
    query = (
        "Người lao động đơn phương chấm dứt "
        "hợp đồng lao động thì phải báo trước bao nhiêu ngày?"
    )

    context = """
[VĂN BẢN]
Bộ luật Lao động

Điều 37
Khoản 3
Phải báo trước ít nhất 45 ngày.
"""

    prompt = build_user_prompt(
        query=query,
        context=context,
    )

    assert query in prompt
    assert context.strip() in prompt
    assert "Trả về JSON" in prompt

    print(
        "\n[PASS] "
        "test_user_prompt_contains_query_and_context"
    )


def main() -> None:
    tests = [
        test_answer_schema_valid,
        test_answer_schema_rejects_extra_fields,
        test_answer_schema_supports_citation,
        test_system_prompt_exists,
        test_user_prompt_contains_query_and_context,
    ]

    print("=" * 80)
    print("ANSWER SCHEMA + PROMPT TEST")
    print("=" * 80)

    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"\n[FAIL] {test.__name__}")
            print(f"Error: {exc}")
            raise

    print("\n" + "=" * 80)
    print(
        f"RESULT: {passed}/{len(tests)} tests passed"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()