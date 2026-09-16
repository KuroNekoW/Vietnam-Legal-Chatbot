import type { AnswerCitation } from "@/lib/api";

interface CitationCardProps {
  citation: AnswerCitation;
  index: number;
}

export default function CitationCard({
  citation,
  index,
}: CitationCardProps) {
  const hasLocation =
    citation.article || citation.clause || citation.point;

  return (
    <details className="citation-card">
      <summary className="citation-summary">
        <span className="citation-index">
          {index}
        </span>

        <span className="citation-title">
          {citation.title}
        </span>
      </summary>

      <div className="citation-body">
        {hasLocation && (
          <div className="citation-location">
            {citation.article && (
              <span>{citation.article}</span>
            )}

            {citation.clause && (
              <span>{citation.clause}</span>
            )}

            {citation.point && (
              <span>{citation.point}</span>
            )}
          </div>
        )}

        {citation.issuance_date && (
          <div className="citation-meta">
            Ngày ban hành: {citation.issuance_date}
          </div>
        )}

        {citation.document_id !== null && (
          <div className="citation-meta">
            Mã văn bản: {citation.document_id}
          </div>
        )}

        {citation.chunk_id && (
          <div className="citation-meta">
            Mã đoạn trích: {citation.chunk_id}
          </div>
        )}
      </div>
    </details>
  );
}
