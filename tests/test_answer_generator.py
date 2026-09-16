from __future__ import annotations

from vn_legal_rag.context import ContextBuilder
from vn_legal_rag.generation import AnswerGenerator


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def make_context():
    builder = ContextBuilder()

    # Reuse the simplest form of context by passing a manually rendered
    # legal context string.
    return builder.build([])


def test_generator_validates_llm_output():
    llm = FakeLLM(
        {
            "answer": (
                "Người lao động làm việc theo hợp đồng lao động "
                "không xác định thời hạn phải báo trước ít nhất 45 ngày."
            ),
            "citations": [
                {
                    "document_id": 123,
                    "title": "Bộ luật Lao động",
                    "article": "Điều 37",
                    "clause": "Khoản 3",
                    "point": None,
                    "issuance_date": "2019-11-20",
                }
            ],
            "insufficient_evidence": False,
            "note": None,
        }
    )

    generator = AnswerGenerator(
        llm=llm
    )

    context = """
[VĂN BẢN]
Bộ luật Lao động

Điều 37. Quyền đơn phương chấm dứt hợp đồng lao động
Khoản 3
Người lao động làm việc theo hợp đồng lao động
không xác định thời hạn phải báo trước ít nhất 45 ngày.
"""

    answer = generator.generate(
        query=(
            "Người lao động đơn phương chấm dứt "
            "hợp đồng lao động thì phải báo trước bao nhiêu ngày?"
        ),
        context=context,
    )

    assert answer.answer
    assert answer.insufficient_evidence is False
    assert len(answer.citations) == 1
    assert answer.citations[0].article == "Điều 37"

    assert len(llm.calls) == 1

    print("\n[PASS] test_generator_validates_llm_output")
    print("\n--- Generated Answer ---")
    print(answer.model_dump_json(indent=2))
    print("--- End ---")


def test_generator_rejects_invalid_llm_output():
    llm = FakeLLM(
        {
            "answer": "",
            "citations": [],
            "insufficient_evidence": False,
            "note": None,
        }
    )

    generator = AnswerGenerator(
        llm=llm
    )

    context = """
[VĂN BẢN]
Bộ luật Lao động

Điều 37
Khoản 3
Phải báo trước ít nhất 45 ngày.
"""

    try:
        generator.generate(
            query="Phải báo trước bao nhiêu ngày?",
            context=context,
        )
    except Exception:
        print(
            "\n[PASS] "
            "test_generator_rejects_invalid_llm_output"
        )
        return

    raise AssertionError(
        "Generator accepted invalid LLM output."
    )


def test_generator_handles_empty_context():
    llm = FakeLLM(
        {}

    )

    generator = AnswerGenerator(
        llm=llm
    )

    answer = generator.generate(
        query="Phải báo trước bao nhiêu ngày?",
        context="",
    )

    assert answer.insufficient_evidence is True
    assert len(llm.calls) == 0

    print(
        "\n[PASS] "
        "test_generator_handles_empty_context"
    )
    print(
        f"  insufficient_evidence: "
        f"{answer.insufficient_evidence}"
    )


def main():
    tests = [
        test_generator_validates_llm_output,
        test_generator_rejects_invalid_llm_output,
        test_generator_handles_empty_context,
    ]

    print("=" * 80)
    print("ANSWER GENERATOR TEST")
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